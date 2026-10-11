#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpSystemContext.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "protocol.h"

namespace {

QNN_INTERFACE_VER_TYPE qnn;
QNN_SYSTEM_INTERFACE_VER_TYPE qsys;
Qnn_LogHandle_t qlog;
Qnn_BackendHandle_t qbackend;
Qnn_DeviceHandle_t qdevice;
QnnHtpDevice_PerfInfrastructure_t* perf;
uint32_t power_id;
uint32_t hw_arch, hw_soc, hw_vtcm_mb;
std::string cache_dir = "/var/cache/synnpu";
std::string build_id;
std::mutex qnn_mutex;
int live_graphs;

void logf(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, "synnpu-qnn: ");
    vfprintf(stderr, fmt, a);
    fputc('\n', stderr);
    va_end(a);
}

void qnn_log(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
    fprintf(stderr, "qnn[%d]: ", int(level));
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

size_t dtype_bits(uint32_t dt) {
    switch (dt & 0xff) {
    case 0x04: return 4;
    case 0x08: return 8;
    case 0x16: return 16;
    case 0x32: return 32;
    case 0x64: return 64;
    }
    return 0;
}

struct Rd {
    const uint8_t* p;
    size_t n, off = 0;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if (off + sizeof(T) > n) { ok = false; return v; }
        memcpy(&v, p + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
    std::string str() {
        uint32_t l = get<uint32_t>();
        if (!ok || off + l > n) { ok = false; return {}; }
        std::string s((const char*)p + off, l);
        off += l;
        return s;
    }
};

struct Wr {
    std::vector<uint8_t> b;
    template <class T> void put(T v) {
        size_t o = b.size();
        b.resize(o + sizeof(T));
        memcpy(b.data() + o, &v, sizeof(T));
    }
    void str(const std::string& s) {
        put<uint32_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
};

struct Mem {
    int fd = -1;
    uint8_t* ptr = nullptr;
    uint64_t size = 0;
    ~Mem() {
        if (ptr) munmap(ptr, size);
        if (fd >= 0) close(fd);
    }
};

struct Tensor {
    uint32_t cid = 0;
    std::string name;
    std::vector<uint32_t> dims;
    std::vector<Qnn_ScaleOffset_t> axis;
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    uint64_t bytes() const {
        uint64_t n = 1;
        for (auto d : dims) n *= d;
        return (n * dtype_bits(t.v1.dataType) + 7) / 8;
    }
    void fix() {
        t.v1.name = name.c_str();
        t.v1.dimensions = dims.data();
        if (t.v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET)
            t.v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset = axis.data();
    }
};

struct Graph {
    Qnn_ContextHandle_t ctx = nullptr;
    Qnn_GraphHandle_t g = nullptr;
    std::vector<std::unique_ptr<Tensor>> ins, outs;
    ~Graph() {
        if (ctx) qnn.contextFree(ctx, nullptr);
    }
};

struct Error {
    uint32_t status;
    std::string msg;
};

[[noreturn]] void fail(uint32_t st, const std::string& m) { throw Error{st, m}; }

void check(Qnn_ErrorHandle_t e, const char* what) {
    if (e != QNN_SUCCESS) fail(SYNNPU_QNN_ERROR, std::string(what) + ": QNN " + std::to_string(e));
}

void power_vote(bool burst) {
    if (!perf) return;
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& d = dcvs.dcvsV3Config;
    d.contextId = power_id;
    d.setDcvsEnable = 1;
    d.dcvsEnable = burst ? 0 : 1;
    d.powerMode = burst ? QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE
                        : QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    d.setSleepLatency = 1;
    d.sleepLatency = burst ? 40 : 1000;
    d.setSleepDisable = 1;
    d.sleepDisable = burst ? 1 : 0;
    d.setBusParams = 1;
    d.setCoreParams = 1;
    auto c = burst ? DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER : DCVS_VOLTAGE_VCORNER_SVS;
    auto lo = burst ? c : DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    d.busVoltageCornerMin = d.coreVoltageCornerMin = lo;
    d.busVoltageCornerTarget = d.coreVoltageCornerTarget = c;
    d.busVoltageCornerMax = d.coreVoltageCornerMax = c;
    QnnHtpPerfInfrastructure_PowerConfig_t rpc = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    rpc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc.rpcControlLatencyConfig = burst ? 100 : 0;
    QnnHtpPerfInfrastructure_PowerConfig_t poll = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    poll.rpcPollingTimeConfig = burst ? 9999 : 0;
    const QnnHtpPerfInfrastructure_PowerConfig_t* cfg[] = {&dcvs, &rpc, &poll, nullptr};
    auto e = perf->setPowerConfig(power_id, cfg);
    if (e != QNN_SUCCESS) logf("setPowerConfig(%s): %lu", burst ? "burst" : "saver", (unsigned long)e);
}

void graph_ref(int delta) {
    int before = live_graphs;
    live_graphs += delta;
    if (before == 0 && live_graphs > 0) power_vote(true);
    if (before > 0 && live_graphs == 0) power_vote(false);
}

void* load(const char* lib) {
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { logf("dlopen %s: %s", lib, dlerror()); exit(1); }
    return h;
}

void init_qnn() {
    void* hb = load("libQnnHtp.so");
    auto get = (decltype(&QnnInterface_getProviders))dlsym(hb, "QnnInterface_getProviders");
    const QnnInterface_t** prov = nullptr;
    uint32_t np = 0;
    if (!get || get(&prov, &np) != QNN_SUCCESS || np == 0) { logf("QnnInterface_getProviders"); exit(1); }
    bool found = false;
    for (uint32_t i = 0; i < np; i++) {
        if (prov[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
            prov[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
            qnn = prov[i]->QNN_INTERFACE_VER_NAME;
            found = true;
            break;
        }
    }
    if (!found) { logf("нет провайдера QNN API %d.%d", QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR); exit(1); }

    void* hs = load("libQnnSystem.so");
    auto gets = (decltype(&QnnSystemInterface_getProviders))dlsym(hs, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** sprov = nullptr;
    if (!gets || gets(&sprov, &np) != QNN_SUCCESS || np == 0) { logf("QnnSystemInterface_getProviders"); exit(1); }
    qsys = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    const char* lv = getenv("SYNNPU_QNN_LOG");
    qnn.logCreate(qnn_log, QnnLog_Level_t(lv ? atoi(lv) : QNN_LOG_LEVEL_WARN), &qlog);
    if (qnn.backendCreate(qlog, nullptr, &qbackend) != QNN_SUCCESS) { logf("backendCreate"); exit(1); }
    const char* bid = nullptr;
    if (qnn.backendGetBuildId(&bid) == QNN_SUCCESS && bid) build_id = bid;

    const QnnDevice_PlatformInfo_t* pi = nullptr;
    if (qnn.deviceGetPlatformInfo(qlog, &pi) == QNN_SUCCESS && pi && pi->v1.numHwDevices > 0) {
        auto ext = (QnnHtpDevice_DeviceInfoExtension_t*)pi->v1.hwDevices[0].v1.deviceInfoExtension;
        if (ext && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
            hw_arch = ext->onChipDevice.arch;
            hw_soc = ext->onChipDevice.socModel;
            hw_vtcm_mb = ext->onChipDevice.vtcmSize;
        }
        qnn.deviceFreePlatformInfo(qlog, pi);
    }
    if (qnn.deviceCreate(qlog, nullptr, &qdevice) != QNN_SUCCESS) { logf("deviceCreate"); exit(1); }

    QnnDevice_Infrastructure_t infra = nullptr;
    if (qnn.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
        auto hi = (QnnHtpDevice_Infrastructure_t*)infra;
        if (hi->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
            perf = &hi->perfInfra;
            if (perf->createPowerConfigId(0, 0, &power_id) != QNN_SUCCESS) perf = nullptr;
        }
    }
    logf("HTP v%u, SoC %u, VTCM %u МБ, QNN %s", hw_arch, hw_soc, hw_vtcm_mb, build_id.c_str());
}

std::string cache_path(const std::string& key) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : key + "|" + build_id + "|" + std::to_string(hw_arch)) h = (h ^ c) * 1099511628211ull;
    char b[32];
    snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return cache_dir + "/" + b + ".bin";
}

std::unique_ptr<Tensor> copy_tensor(const Qnn_Tensor_t& src) {
    auto t = std::make_unique<Tensor>();
    const Qnn_TensorV1_t& s = src.v1;
    t->name = s.name ? s.name : "";
    t->dims.assign(s.dimensions, s.dimensions + s.rank);
    t->t.v1.id = s.id;
    t->t.v1.type = s.type;
    t->t.v1.dataFormat = s.dataFormat;
    t->t.v1.dataType = s.dataType;
    t->t.v1.quantizeParams = s.quantizeParams;
    if (s.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        auto& a = s.quantizeParams.axisScaleOffsetEncoding;
        t->axis.assign(a.scaleOffset, a.scaleOffset + a.numScaleOffsets);
    }
    t->t.v1.rank = s.rank;
    t->t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t->t.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
    if (t->name.size() > 1 && t->name[0] == 't') t->cid = strtoul(t->name.c_str() + 1, nullptr, 10);
    t->fix();
    return t;
}

struct Group {
    Qnn_ContextHandle_t first = nullptr;
    uint64_t cap = 0;
};

uint64_t spill_fill_of(const QnnSystemContext_BinaryInfo_t* info, const QnnSystemContext_GraphInfo_t& g) {
    if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 && g.graphInfoV3.graphBlobInfo) {
        auto b = (const QnnHtpSystemContext_GraphBlobInfo_t*)g.graphInfoV3.graphBlobInfo;
        if (b->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1)
            return b->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
    }
    const void* hw = nullptr;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) hw = info->contextBinaryInfoV1.hwInfoBlob;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) hw = info->contextBinaryInfoV2.hwInfoBlob;
    if (hw) {
        auto b = (const QnnHtpSystemContext_HwBlobInfo_t*)hw;
        if (b->version == QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1)
            return b->contextBinaryHwInfoBlobV1_t.spillFillBufferSize;
    }
    return 0;
}

std::unique_ptr<Graph> load_cached(const std::string& path, Group* group = nullptr) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return nullptr;
    struct stat st;
    fstat(fd, &st);
    std::vector<uint8_t> buf(st.st_size);
    size_t got = 0;
    while (got < buf.size()) {
        ssize_t r = read(fd, buf.data() + got, buf.size() - got);
        if (r <= 0) break;
        got += r;
    }
    close(fd);
    if (got != buf.size()) return nullptr;

    QnnSystemContext_Handle_t sc = nullptr;
    check(qsys.systemContextCreate(&sc), "systemContextCreate");
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t isz = 0;
    auto e = qsys.systemContextGetBinaryInfo(sc, buf.data(), buf.size(), &info, &isz);
    if (e != QNN_SUCCESS || !info) {
        qsys.systemContextFree(sc);
        logf("кэш %s не читается (%lu) — пересборка", path.c_str(), (unsigned long)e);
        return nullptr;
    }
    uint32_t ng = 0;
    QnnSystemContext_GraphInfo_t* gs = nullptr;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        ng = info->contextBinaryInfoV3.numGraphs;
        gs = info->contextBinaryInfoV3.graphs;
    } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        ng = info->contextBinaryInfoV2.numGraphs;
        gs = info->contextBinaryInfoV2.graphs;
    } else {
        ng = info->contextBinaryInfoV1.numGraphs;
        gs = info->contextBinaryInfoV1.graphs;
    }
    auto g = std::make_unique<Graph>();
    if (ng != 1) {
        qsys.systemContextFree(sc);
        fail(SYNNPU_IO_ERROR, "в кэше не один граф");
    }
    uint64_t spill = spill_fill_of(info, gs[0]);
    auto& gi = gs[0].graphInfoV1;
    for (uint32_t i = 0; i < gi.numGraphInputs; i++) g->ins.push_back(copy_tensor(gi.graphInputs[i]));
    for (uint32_t i = 0; i < gi.numGraphOutputs; i++) g->outs.push_back(copy_tensor(gi.graphOutputs[i]));
    std::string gname = gi.graphName;
    qsys.systemContextFree(sc);

    QnnHtpContext_CustomConfig_t gr{};
    gr.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    QnnContext_Config_t gc = QNN_CONTEXT_CONFIG_INIT;
    gc.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    gc.customConfig = &gr;
    const QnnContext_Config_t* cfg[] = {&gc, nullptr};
    bool grouped = group && spill > 0 && (!group->first || spill <= group->cap);
    if (grouped) {
        if (!group->first) {
            const char* mb = getenv("SYNNPU_SPILLFILL_MB");
            group->cap = std::max<uint64_t>(spill, uint64_t(mb ? atoi(mb) : 256) << 20);
        }
        gr.groupRegistration.firstGroupHandle = group->first;
        gr.groupRegistration.maxSpillFillBuffer = group->cap;
    }
    logf("контекст %s: %.1f МБ, spill-fill %.1f МБ%s", path.c_str(), buf.size() / 1048576.0, spill / 1048576.0,
         grouped ? (group->first ? " (общий буфер группы)" : " (новая группа)") : "");
    check(qnn.contextCreateFromBinary(qbackend, qdevice, grouped ? cfg : nullptr, buf.data(), buf.size(), &g->ctx, nullptr),
          "contextCreateFromBinary");
    check(qnn.graphRetrieve(g->ctx, gname.c_str(), &g->g), "graphRetrieve");
    if (grouped && !group->first) group->first = g->ctx;
    return g;
}

void save_cache(const std::string& path, Qnn_ContextHandle_t ctx) {
    Qnn_ContextBinarySize_t sz = 0;
    check(qnn.contextGetBinarySize(ctx, &sz), "contextGetBinarySize");
    std::vector<uint8_t> buf(sz);
    Qnn_ContextBinarySize_t wr = 0;
    check(qnn.contextGetBinary(ctx, buf.data(), sz, &wr), "contextGetBinary");
    mkdir(cache_dir.c_str(), 0755);
    std::string tmp = path + ".tmp." + std::to_string(getpid()) + "." + std::to_string(gettid());
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { logf("кэш %s: %s", tmp.c_str(), strerror(errno)); return; }
    size_t off = 0;
    while (off < wr) {
        ssize_t r = write(fd, buf.data() + off, wr - off);
        if (r <= 0) break;
        off += r;
    }
    close(fd);
    if (off == wr) rename(tmp.c_str(), path.c_str());
    else unlink(tmp.c_str());
}

struct Session {
    int sock;
    std::unordered_map<uint32_t, std::unique_ptr<Mem>> mems;
    std::unordered_map<uint32_t, std::unique_ptr<Graph>> graphs;
    uint32_t next_mem = 1, next_graph = 1;
    Group group;
    bool prepare_only = false;

    ~Session() {
        std::lock_guard<std::mutex> l(qnn_mutex);
        if (!graphs.empty()) graph_ref(-int(graphs.size()));
        graphs.clear();
    }

    Mem& mem(uint32_t id) {
        auto it = mems.find(id);
        if (it == mems.end()) fail(SYNNPU_BAD_REQUEST, "нет памяти " + std::to_string(id));
        return *it->second;
    }

    uint8_t* span(uint32_t id, uint64_t off, uint64_t bytes) {
        Mem& m = mem(id);
        if (off > m.size || bytes > m.size - off) fail(SYNNPU_BAD_REQUEST, "выход за границы памяти " + std::to_string(id));
        return m.ptr + off;
    }

    void io_table(Wr& w, Graph& g) {
        for (auto* v : {&g.ins, &g.outs}) {
            w.put<uint32_t>(v->size());
            for (auto& t : *v) {
                w.put<uint32_t>(t->cid);
                w.put<uint32_t>(t->t.v1.dataType);
                w.put<uint32_t>(t->dims.size());
                for (auto d : t->dims) w.put<uint32_t>(d);
            }
        }
    }

    uint32_t add_graph(std::unique_ptr<Graph> g) {
        uint32_t id = next_graph++;
        graphs[id] = std::move(g);
        graph_ref(1);
        return id;
    }

    void hello(Rd&, Wr& w) {
        w.put<uint32_t>(SYNNPU_VERSION);
        w.put<uint32_t>(hw_arch);
        w.put<uint32_t>(hw_soc);
        w.put<uint32_t>(hw_vtcm_mb);
        w.str(build_id);
    }

    void mem_register(Rd& r, Wr& w, int fd) {
        uint64_t size = r.get<uint64_t>();
        if (!r.ok || fd < 0 || size == 0) {
            if (fd >= 0) close(fd);
            fail(SYNNPU_BAD_REQUEST, "MEM_REGISTER: нужны fd и размер");
        }
        auto m = std::make_unique<Mem>();
        m->fd = fd;
        m->size = size;
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) fail(SYNNPU_IO_ERROR, std::string("mmap: ") + strerror(errno));
        m->ptr = (uint8_t*)p;
        uint32_t id = next_mem++;
        mems[id] = std::move(m);
        w.put<uint32_t>(id);
    }

    void mem_free(Rd& r, Wr&) {
        uint32_t id = r.get<uint32_t>();
        if (!mems.erase(id)) fail(SYNNPU_BAD_REQUEST, "нет памяти " + std::to_string(id));
    }

    void graph_open(Rd& r, Wr& w) {
        std::string key = r.str();
        if (!r.ok) fail(SYNNPU_BAD_REQUEST, "GRAPH_OPEN: ключ");
        std::lock_guard<std::mutex> l(qnn_mutex);
        auto g = load_cached(cache_path(key), &group);
        if (!g) fail(SYNNPU_NOT_FOUND, "нет в кэше");
        Graph& gr = *g;
        w.put<uint32_t>(add_graph(std::move(g)));
        io_table(w, gr);
    }

    void graph_build(Rd& r, Wr& w) {
        auto t0 = std::chrono::steady_clock::now();
        std::string key = r.str();
        if (!r.ok) fail(SYNNPU_BAD_REQUEST, "GRAPH_BUILD: заголовок");
        mkdir(cache_dir.c_str(), 0755);
        std::string req = cache_dir + "/req." + std::to_string(getpid()) + "." + std::to_string(gettid());
        int fd = open(req.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) fail(SYNNPU_IO_ERROR, req + ": " + strerror(errno));
        size_t off = 0;
        while (off < r.n) {
            ssize_t k = write(fd, r.p + off, r.n - off);
            if (k <= 0) break;
            off += k;
        }
        close(fd);
        if (off != r.n) {
            unlink(req.c_str());
            fail(SYNNPU_IO_ERROR, "запрос сборки не записан");
        }
        std::vector<std::string> args = {"synnpu-qnn", "--build", req};
        std::vector<int> fds;
        for (auto& [id, m] : mems) {
            args.push_back(std::to_string(id) + ":" + std::to_string(m->fd) + ":" + std::to_string(m->size));
            fds.push_back(m->fd);
        }
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        pid_t pid = fork();
        if (pid == 0) {
            for (int f : fds) fcntl(f, F_SETFD, 0);
            execv("/proc/self/exe", argv.data());
            _exit(127);
        }
        int st = 0;
        if (pid < 0) {
            unlink(req.c_str());
            fail(SYNNPU_IO_ERROR, std::string("fork: ") + strerror(errno));
        }
        for (;;) {
            pid_t r = waitpid(pid, &st, WNOHANG);
            if (r == pid) break;
            if (r < 0 && errno != EINTR) {
                unlink(req.c_str());
                fail(SYNNPU_IO_ERROR, std::string("waitpid: ") + strerror(errno));
            }
            pollfd pf{sock, POLLRDHUP, 0};
            if (poll(&pf, 1, 200) > 0 && (pf.revents & (POLLRDHUP | POLLHUP | POLLERR))) {
                kill(pid, SIGKILL);
                waitpid(pid, &st, 0);
                unlink(req.c_str());
                logf("клиент отключился — сборка графа остановлена");
                fail(SYNNPU_IO_ERROR, "клиент отключился");
            }
        }
        unlink(req.c_str());
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            fail(SYNNPU_QNN_ERROR, "сборка графа в процессе " + std::to_string(pid) + " не удалась (" +
                                       (WIFSIGNALED(st) ? "сигнал " + std::to_string(WTERMSIG(st))
                                                        : "код " + std::to_string(WEXITSTATUS(st))) + ")");
        }
        std::lock_guard<std::mutex> l(qnn_mutex);
        auto g = load_cached(cache_path(key), &group);
        if (!g) fail(SYNNPU_IO_ERROR, "собранный граф не найден в кэше");
        uint32_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        Graph& gr = *g;
        w.put<uint32_t>(add_graph(std::move(g)));
        io_table(w, gr);
        w.put<uint32_t>(ms);
    }

    void build_inproc(Rd& r, Wr& w) {
        auto t0 = std::chrono::steady_clock::now();
        std::string key = r.str();
        uint32_t flags = r.get<uint32_t>();
        uint32_t nt = r.get<uint32_t>();
        if (!r.ok) fail(SYNNPU_BAD_REQUEST, "GRAPH_BUILD: заголовок");

        std::lock_guard<std::mutex> l(qnn_mutex);
        auto g = std::make_unique<Graph>();
        QnnHtpContext_CustomConfig_t po{};
        po.option = QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY;
        po.isPrepareOnly = true;
        QnnContext_Config_t pc = QNN_CONTEXT_CONFIG_INIT;
        pc.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
        pc.customConfig = &po;
        const QnnContext_Config_t* pcfg[] = {&pc, nullptr};
        check(qnn.contextCreate(qbackend, qdevice, prepare_only ? pcfg : nullptr, &g->ctx), "contextCreate");

        QnnHtpGraph_CustomConfig_t prec = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        prec.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        prec.precision = QNN_PRECISION_FLOAT16;
        QnnHtpGraph_CustomConfig_t opt = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        opt.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        opt.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        uint32_t olvl = (flags >> 8) & 0xf;
        opt.optimizationOption.floatValue = olvl ? float(olvl) : (getenv("SYNNPU_OPT") ? float(atoi(getenv("SYNNPU_OPT"))) : 2.0f);
        QnnHtpGraph_CustomConfig_t vtcm = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        vtcm.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        vtcm.vtcmSizeInMB = hw_vtcm_mb ? hw_vtcm_mb : 8;
        std::vector<QnnGraph_Config_t> gc;
        for (auto* c : {&opt, &vtcm, &prec}) {
            if (c == &prec && !(flags & SYNNPU_BUILD_FP16)) continue;
            QnnGraph_Config_t x = QNN_GRAPH_CONFIG_INIT;
            x.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            x.customConfig = c;
            gc.push_back(x);
        }
        std::vector<const QnnGraph_Config_t*> gcp;
        for (auto& x : gc) gcp.push_back(&x);
        gcp.push_back(nullptr);
        check(qnn.graphCreate(g->ctx, "g", gcp.data(), &g->g), "graphCreate");

        std::unordered_map<uint32_t, std::unique_ptr<Tensor>> all;
        for (uint32_t i = 0; i < nt; i++) {
            auto t = std::make_unique<Tensor>();
            t->cid = r.get<uint32_t>();
            uint8_t kind = r.get<uint8_t>();
            uint32_t dt = r.get<uint32_t>();
            uint8_t rank = r.get<uint8_t>();
            for (uint8_t k = 0; k < rank; k++) t->dims.push_back(r.get<uint32_t>());
            uint8_t q = r.get<uint8_t>();
            auto& qp = t->t.v1.quantizeParams;
            if (q == SYNNPU_QUANT_SCALE_OFFSET) {
                qp.encodingDefinition = QNN_DEFINITION_DEFINED;
                qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                qp.scaleOffsetEncoding.scale = r.get<float>();
                qp.scaleOffsetEncoding.offset = r.get<int32_t>();
            } else if (q == SYNNPU_QUANT_AXIS_SCALE_OFFSET) {
                qp.encodingDefinition = QNN_DEFINITION_DEFINED;
                qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
                qp.axisScaleOffsetEncoding.axis = r.get<int32_t>();
                uint32_t n = r.get<uint32_t>();
                if (!r.ok || n > r.n) fail(SYNNPU_BAD_REQUEST, "квантование по оси");
                for (uint32_t k = 0; k < n; k++) {
                    Qnn_ScaleOffset_t so;
                    so.scale = r.get<float>();
                    so.offset = r.get<int32_t>();
                    t->axis.push_back(so);
                }
                qp.axisScaleOffsetEncoding.numScaleOffsets = n;
            } else if (q != SYNNPU_QUANT_NONE) {
                fail(SYNNPU_BAD_REQUEST, "неизвестное квантование");
            }
            t->name = "t" + std::to_string(t->cid);
            auto& v = t->t.v1;
            v.dataType = Qnn_DataType_t(dt);
            v.rank = rank;
            v.memType = QNN_TENSORMEMTYPE_RAW;
            switch (kind) {
            case SYNNPU_INPUT: v.type = QNN_TENSOR_TYPE_APP_WRITE; break;
            case SYNNPU_OUTPUT: v.type = QNN_TENSOR_TYPE_APP_READ; break;
            case SYNNPU_STATIC: v.type = QNN_TENSOR_TYPE_STATIC; break;
            case SYNNPU_NATIVE: v.type = QNN_TENSOR_TYPE_NATIVE; break;
            default: fail(SYNNPU_BAD_REQUEST, "неизвестный вид тензора");
            }
            if (kind == SYNNPU_STATIC) {
                uint32_t mid = r.get<uint32_t>();
                uint64_t off = r.get<uint64_t>();
                uint64_t bytes = r.get<uint64_t>();
                if (!r.ok) fail(SYNNPU_BAD_REQUEST, "данные статического тензора");
                if (bytes != t->bytes()) fail(SYNNPU_BAD_REQUEST, t->name + ": размер данных не совпадает с формой");
                v.clientBuf.data = span(mid, off, bytes);
                v.clientBuf.dataSize = bytes;
            }
            if (!r.ok || !dtype_bits(dt)) fail(SYNNPU_BAD_REQUEST, "описание тензора " + std::to_string(i));
            t->fix();
            check(qnn.tensorCreateGraphTensor(g->g, &t->t), ("tensorCreateGraphTensor " + t->name).c_str());
            all[t->cid] = std::move(t);
        }
        auto ref = [&](uint32_t id) -> Tensor& {
            auto it = all.find(id);
            if (it == all.end()) fail(SYNNPU_BAD_REQUEST, "нет тензора " + std::to_string(id));
            return *it->second;
        };

        uint32_t nn = r.get<uint32_t>();
        for (uint32_t i = 0; i < nn && r.ok; i++) {
            std::string name = r.str(), pkg = r.str(), type = r.str();
            std::vector<Qnn_Tensor_t> in, out;
            uint32_t ni = r.get<uint32_t>();
            for (uint32_t k = 0; k < ni && r.ok; k++) in.push_back(ref(r.get<uint32_t>()).t);
            uint32_t no = r.get<uint32_t>();
            for (uint32_t k = 0; k < no && r.ok; k++) out.push_back(ref(r.get<uint32_t>()).t);
            uint32_t np = r.get<uint32_t>();
            std::vector<std::string> pnames;
            std::vector<Qnn_Param_t> params;
            pnames.reserve(np);
            for (uint32_t k = 0; k < np && r.ok; k++) {
                pnames.push_back(r.str());
                Qnn_Param_t p = QNN_PARAM_INIT;
                uint8_t pk = r.get<uint8_t>();
                if (pk == SYNNPU_PARAM_SCALAR) {
                    p.paramType = QNN_PARAMTYPE_SCALAR;
                    p.scalarParam.dataType = Qnn_DataType_t(r.get<uint32_t>());
                    uint64_t raw = r.get<uint64_t>();
                    memcpy(&p.scalarParam.uint64Value, &raw, sizeof raw);
                } else if (pk == SYNNPU_PARAM_TENSOR) {
                    p.paramType = QNN_PARAMTYPE_TENSOR;
                    p.tensorParam = ref(r.get<uint32_t>()).t;
                } else {
                    fail(SYNNPU_BAD_REQUEST, "неизвестный вид параметра");
                }
                params.push_back(p);
            }
            if (!r.ok) break;
            for (uint32_t k = 0; k < np; k++) params[k].name = pnames[k].c_str();
            Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
            op.v1.name = name.c_str();
            op.v1.packageName = pkg.c_str();
            op.v1.typeName = type.c_str();
            op.v1.numOfParams = np;
            op.v1.params = params.data();
            op.v1.numOfInputs = ni;
            op.v1.inputTensors = in.data();
            op.v1.numOfOutputs = no;
            op.v1.outputTensors = out.data();
            check(qnn.graphAddNode(g->g, op), ("graphAddNode " + name + " (" + type + ")").c_str());
        }
        if (!r.ok) fail(SYNNPU_BAD_REQUEST, "GRAPH_BUILD: описание узлов");
        check(qnn.graphFinalize(g->g, nullptr, nullptr), "graphFinalize");

        for (auto& [id, t] : all) {
            auto ty = t->t.v1.type;
            if (ty == QNN_TENSOR_TYPE_APP_WRITE || ty == QNN_TENSOR_TYPE_APP_READ) {
                auto c = copy_tensor(t->t);
                c->cid = id;
                (ty == QNN_TENSOR_TYPE_APP_WRITE ? g->ins : g->outs).push_back(std::move(c));
            }
        }
        save_cache(cache_path(key), g->ctx);
        uint32_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        logf("граф «%s»: %u тензоров, %u узлов, сборка %u мс", key.c_str(), nt, nn, ms);
        Graph& gr = *g;
        w.put<uint32_t>(add_graph(std::move(g)));
        io_table(w, gr);
        w.put<uint32_t>(ms);
    }

    void graph_free(Rd& r, Wr&) {
        uint32_t id = r.get<uint32_t>();
        std::lock_guard<std::mutex> l(qnn_mutex);
        auto it = graphs.find(id);
        if (it == graphs.end()) fail(SYNNPU_BAD_REQUEST, "нет графа " + std::to_string(id));
        if (it->second->ctx == group.first) group = Group{};
        graphs.erase(it);
        graph_ref(-1);
    }

    void execute(Rd& r, Wr& w) {
        uint32_t gid = r.get<uint32_t>();
        uint32_t nb = r.get<uint32_t>();
        auto it = graphs.find(gid);
        if (!r.ok || it == graphs.end()) fail(SYNNPU_BAD_REQUEST, "нет графа " + std::to_string(gid));
        Graph& g = *it->second;
        std::unordered_map<uint32_t, std::pair<uint32_t, uint64_t>> bind;
        for (uint32_t i = 0; i < nb; i++) {
            uint32_t tid = r.get<uint32_t>();
            uint32_t mid = r.get<uint32_t>();
            uint64_t off = r.get<uint64_t>();
            bind[tid] = {mid, off};
        }
        if (!r.ok) fail(SYNNPU_BAD_REQUEST, "EXECUTE: привязки");
        std::vector<Qnn_Tensor_t> in, out;
        for (auto* v : {&g.ins, &g.outs}) {
            for (auto& t : *v) {
                auto b = bind.find(t->cid);
                if (b == bind.end()) fail(SYNNPU_BAD_REQUEST, "не привязан тензор " + std::to_string(t->cid));
                Qnn_Tensor_t x = t->t;
                uint64_t bytes = t->bytes();
                x.v1.memType = QNN_TENSORMEMTYPE_RAW;
                x.v1.clientBuf.data = span(b->second.first, b->second.second, bytes);
                x.v1.clientBuf.dataSize = bytes;
                (v == &g.ins ? in : out).push_back(x);
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        check(qnn.graphExecute(g.g, in.data(), in.size(), out.data(), out.size(), nullptr, nullptr), "graphExecute");
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        w.put<uint64_t>(us);
    }
};

bool read_full(int fd, void* p, size_t n) {
    auto* b = (uint8_t*)p;
    while (n) {
        ssize_t r = read(fd, b, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        b += r;
        n -= r;
    }
    return true;
}

bool write_full(int fd, const void* p, size_t n) {
    auto* b = (const uint8_t*)p;
    while (n) {
        ssize_t r = send(fd, b, n, MSG_NOSIGNAL);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        b += r;
        n -= r;
    }
    return true;
}

void serve(int sock) {
    Session s{sock};
    for (;;) {
        synnpu_hdr h;
        int fd = -1;
        char cbuf[CMSG_SPACE(sizeof(int))];
        iovec iov{&h, sizeof h};
        msghdr m{};
        m.msg_iov = &iov;
        m.msg_iovlen = 1;
        m.msg_control = cbuf;
        m.msg_controllen = sizeof cbuf;
        ssize_t got = recvmsg(sock, &m, MSG_CMSG_CLOEXEC);
        if (got <= 0) break;
        for (cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
            if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) memcpy(&fd, CMSG_DATA(c), sizeof fd);
        if (size_t(got) < sizeof h && !read_full(sock, (uint8_t*)&h + got, sizeof h - got)) break;
        if (h.magic != SYNNPU_MAGIC || h.len > (1ull << 32)) { logf("плохой заголовок"); break; }
        std::vector<uint8_t> body(h.len);
        if (h.len && !read_full(sock, body.data(), h.len)) break;

        Rd r{body.data(), body.size()};
        Wr w;
        uint32_t status = SYNNPU_OK;
        std::string msg;
        try {
            switch (h.op) {
            case SYNNPU_HELLO: s.hello(r, w); break;
            case SYNNPU_MEM_REGISTER: s.mem_register(r, w, fd); fd = -1; break;
            case SYNNPU_MEM_FREE: s.mem_free(r, w); break;
            case SYNNPU_GRAPH_OPEN: s.graph_open(r, w); break;
            case SYNNPU_GRAPH_BUILD: s.graph_build(r, w); break;
            case SYNNPU_GRAPH_FREE: s.graph_free(r, w); break;
            case SYNNPU_EXECUTE: s.execute(r, w); break;
            default: fail(SYNNPU_BAD_REQUEST, "неизвестная операция " + std::to_string(h.op));
            }
        } catch (const Error& e) {
            status = e.status;
            msg = e.msg;
            w.b.clear();
            if (status != SYNNPU_NOT_FOUND) logf("op %u: %s", h.op, msg.c_str());
        }
        if (fd >= 0) close(fd);
        Wr out;
        out.put<uint32_t>(status);
        out.str(msg);
        out.b.insert(out.b.end(), w.b.begin(), w.b.end());
        synnpu_hdr rh{SYNNPU_MAGIC, h.op | SYNNPU_REPLY, h.seq, 0, out.b.size()};
        if (!write_full(sock, &rh, sizeof rh) || !write_full(sock, out.b.data(), out.b.size())) break;
    }
    close(sock);
}

}

int build_child(int argc, char** argv) {
    init_qnn();
    Session s{-1};
    s.prepare_only = !getenv("SYNNPU_NO_PREPARE_ONLY");
    for (int i = 3; i < argc; i++) {
        unsigned id = 0, fd = 0;
        unsigned long long size = 0;
        if (sscanf(argv[i], "%u:%u:%llu", &id, &fd, &size) != 3) return 2;
        auto m = std::make_unique<Mem>();
        m->fd = int(fd);
        m->size = size;
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m->fd, 0);
        if (p == MAP_FAILED) { logf("mmap %u: %s", id, strerror(errno)); return 3; }
        m->ptr = (uint8_t*)p;
        s.mems[id] = std::move(m);
    }
    int fd = open(argv[2], O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 4;
    struct stat st;
    fstat(fd, &st);
    std::vector<uint8_t> body(st.st_size);
    if (!read_full(fd, body.data(), body.size())) return 5;
    close(fd);
    Rd r{body.data(), body.size()};
    Wr w;
    int rc = 0;
    try {
        s.build_inproc(r, w);
    } catch (const Error& e) {
        logf("сборка: %s", e.msg.c_str());
        rc = 1;
    }
    FILE* f = fopen("/proc/self/status", "r");
    char line[256];
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "VmHWM:", 6)) logf("сборка: пик памяти %s", line + 6);
    if (f) fclose(f);
    return rc;
}

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    setvbuf(stderr, nullptr, _IOLBF, 0);
    if (const char* c = getenv("SYNNPU_CACHE")) cache_dir = c;
    if (argc >= 3 && strcmp(argv[1], "--build") == 0) _exit(build_child(argc, argv));
    init_qnn();

    const char* path = getenv("SYNNPU_SOCKET") ? getenv("SYNNPU_SOCKET") : SYNNPU_SOCKET;
    std::string dir(path);
    dir.resize(dir.rfind('/'));
    mkdir(dir.c_str(), 0755);
    unlink(path);
    int ls = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    strncpy(a.sun_path, path, sizeof a.sun_path - 1);
    if (bind(ls, (sockaddr*)&a, sizeof a) < 0 || listen(ls, 16) < 0) {
        logf("сокет %s: %s", path, strerror(errno));
        return 1;
    }
    gid_t gid = getenv("SYNNPU_GID") ? gid_t(atoi(getenv("SYNNPU_GID"))) : gid_t(-1);
    if (gid != gid_t(-1)) {
        chown(dir.c_str(), 0, gid);
        chown(path, 0, gid);
    }
    chmod(dir.c_str(), gid != gid_t(-1) ? 0750 : 0755);
    chmod(path, gid != gid_t(-1) ? 0660 : 0666);
    logf("слушаю %s", path);
    for (;;) {
        int c = accept4(ls, nullptr, nullptr, SOCK_CLOEXEC);
        if (c < 0) {
            if (errno == EINTR) continue;
            logf("accept: %s", strerror(errno));
            continue;
        }
        std::thread(serve, c).detach();
    }
}
