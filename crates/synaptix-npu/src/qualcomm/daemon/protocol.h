#pragma once
#include <stdint.h>

#define SYNNPU_SOCKET "/run/synnpu/qnn.sock"
#define SYNNPU_MAGIC 0x55504e53u
#define SYNNPU_VERSION 1u
#define SYNNPU_REPLY 0x80000000u

enum synnpu_op {
    SYNNPU_HELLO = 1,
    SYNNPU_MEM_REGISTER = 2,
    SYNNPU_MEM_FREE = 3,
    SYNNPU_GRAPH_OPEN = 4,
    SYNNPU_GRAPH_BUILD = 5,
    SYNNPU_GRAPH_FREE = 6,
    SYNNPU_EXECUTE = 7,
};

enum synnpu_status {
    SYNNPU_OK = 0,
    SYNNPU_NOT_FOUND = 1,
    SYNNPU_BAD_REQUEST = 2,
    SYNNPU_QNN_ERROR = 3,
    SYNNPU_IO_ERROR = 4,
};

enum synnpu_tensor_kind {
    SYNNPU_INPUT = 0,
    SYNNPU_OUTPUT = 1,
    SYNNPU_STATIC = 2,
    SYNNPU_NATIVE = 3,
};

enum synnpu_quant {
    SYNNPU_QUANT_NONE = 0,
    SYNNPU_QUANT_SCALE_OFFSET = 1,
    SYNNPU_QUANT_AXIS_SCALE_OFFSET = 2,
};

enum synnpu_param_kind {
    SYNNPU_PARAM_SCALAR = 0,
    SYNNPU_PARAM_TENSOR = 1,
};

#define SYNNPU_BUILD_FP16 1u

struct synnpu_hdr {
    uint32_t magic;
    uint32_t op;
    uint32_t seq;
    uint32_t reserved;
    uint64_t len;
};
