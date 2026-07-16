//! B3: RadixTree release (ref-count + эвикция) — аналитически.

use synaptix_infer::memory::radix_tree::RadixTree;

#[test]
fn t40_1_insert_counts_actual_nodes() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3], 100);
    assert_eq!(tree.total_nodes, 3);
    // Общий префикс [1,2] не создаёт новых узлов.
    tree.insert(&[1, 2, 4], 200);
    assert_eq!(tree.total_nodes, 4, "shared prefix [1,2] reused, only node 4 added");

    let (m, v) = tree.lookup(&[1, 2, 3]);
    assert_eq!((m, v), (3, Some(100)));
    let (m, v) = tree.lookup(&[1, 2, 4]);
    assert_eq!((m, v), (3, Some(200)));
}

#[test]
fn t40_2_release_evicts_childless_zero_ref() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3], 100);
    tree.insert(&[1, 2, 4], 200);
    assert_eq!(tree.total_nodes, 4);

    // Отпускаем [1,2,3]: узел 3 (ref→0, без детей) вытесняется; [1,2] живут.
    let removed = tree.release(&[1, 2, 3]);
    assert_eq!(removed, 1);
    assert_eq!(tree.total_nodes, 3);

    // [1,2,4] всё ещё доступен; [1,2,3] — нет (value у узла 2 отсутствует).
    assert_eq!(tree.lookup(&[1, 2, 4]), (3, Some(200)));
    assert_eq!(tree.lookup(&[1, 2, 3]), (2, None));

    // Отпускаем [1,2,4]: каскад 4→2→1 (все ref→0, childless).
    let removed = tree.release(&[1, 2, 4]);
    assert_eq!(removed, 3);
    assert_eq!(tree.total_nodes, 0);
    assert_eq!(tree.lookup(&[1, 2, 4]), (0, None));
}

#[test]
fn t40_3_release_keeps_shared_refs() {
    let mut tree = RadixTree::new();
    // Дважды вставляем один и тот же путь — ref_count==2 у каждого узла.
    tree.insert(&[7, 8], 1);
    tree.insert(&[7, 8], 2);
    assert_eq!(tree.total_nodes, 2);

    // Первый release не вытесняет (ещё одна ссылка держит путь).
    let removed = tree.release(&[7, 8]);
    assert_eq!(removed, 0);
    assert_eq!(tree.total_nodes, 2);
    assert_eq!(tree.lookup(&[7, 8]).0, 2);

    // Второй release вытесняет оба узла.
    let removed = tree.release(&[7, 8]);
    assert_eq!(removed, 2);
    assert_eq!(tree.total_nodes, 0);
}

#[test]
fn t40_4_release_unknown_path_noop() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2], 1);
    assert_eq!(tree.release(&[9, 9, 9]), 0);
    assert_eq!(tree.release(&[1, 5]), 0, "partial path not present");
    assert_eq!(tree.total_nodes, 2);
}
