#[macro_export]
macro_rules! shape_assert {
    ($t:expr, $expected:expr $(,)?) => {{
        let actual = $t.dims();
        let expected: &[usize] = &$expected;
        if actual != expected {
            panic!(
                "shape_assert!({}): expected {:?}, got {:?}",
                stringify!($t),
                expected,
                actual
            );
        }
    }};
    ($t:expr, $expected:expr, $label:expr $(,)?) => {{
        let actual = $t.dims();
        let expected: &[usize] = &$expected;
        if actual != expected {
            panic!(
                "shape_assert!({}, {}): expected {:?}, got {:?}",
                stringify!($t),
                $label,
                expected,
                actual
            );
        }
    }};
}

#[macro_export]
macro_rules! shape_assert_rank {
    ($t:expr, $rank:expr $(,)?) => {{
        let actual = $t.rank();
        let expected: usize = $rank;
        if actual != expected {
            panic!(
                "shape_assert_rank!({}): expected rank {}, got {} (dims {:?})",
                stringify!($t),
                expected,
                actual,
                $t.dims()
            );
        }
    }};
}

#[macro_export]
macro_rules! shape_assert_eq {
    ($a:expr, $b:expr $(,)?) => {{
        let da = $a.dims();
        let db = $b.dims();
        if da != db {
            panic!(
                "shape_assert_eq!({}, {}): {:?} != {:?}",
                stringify!($a),
                stringify!($b),
                da,
                db
            );
        }
    }};
}
