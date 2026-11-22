//! Which logical CPUs share a core, for the threads that keep several cores busy (the copy
//! workers of [`copy_parallel`](super::copy_parallel), the workers of the CPU backend). Linux
//! reads the sysfs topology; elsewhere nothing is known and every query says so.

/// The logical CPU the calling thread runs on right now, if the platform can tell.
#[cfg(target_os = "linux")]
pub fn current_cpu() -> Option<usize> {
    // SAFETY: `sched_getcpu` has no preconditions; it returns -1 if it cannot tell.
    usize::try_from(unsafe { libc::sched_getcpu() }).ok()
}

#[cfg(not(target_os = "linux"))]
pub fn current_cpu() -> Option<usize> {
    None
}

/// Whether the two distinct logical CPUs are hardware threads of one core (SMT siblings), which
/// then share the core's execution units, caches and memory bandwidth. `false` if unknown.
#[cfg(target_os = "linux")]
pub fn shares_core(a: usize, b: usize) -> bool {
    a != b && imp::siblings().get(a).is_some_and(|list| list.contains(&b))
}

#[cfg(not(target_os = "linux"))]
pub fn shares_core(_: usize, _: usize) -> bool {
    false
}

/// The physical cores the process may run on (logical CPUs of one core count once), or `None` if
/// the topology cannot be read.
#[cfg(target_os = "linux")]
pub fn physical_cores() -> Option<usize> {
    imp::physical_cores()
}

#[cfg(not(target_os = "linux"))]
pub fn physical_cores() -> Option<usize> {
    None
}

#[cfg(target_os = "linux")]
mod imp {
    use std::sync::LazyLock;

    /// The logical CPUs of every logical CPU's core (itself included), by CPU number; empty for a
    /// CPU whose topology is not readable (offline, or no sysfs).
    pub(super) fn siblings() -> &'static [Vec<usize>] {
        static SIBLINGS: LazyLock<Vec<Vec<usize>>> = LazyLock::new(|| {
            // SAFETY: `sysconf` has no preconditions.
            let configured = usize::try_from(unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) }).unwrap_or(0);
            (0..configured)
                .map(|cpu| {
                    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))
                        .map(|text| parse_list(&text))
                        .unwrap_or_default()
                })
                .collect()
        });
        &SIBLINGS
    }

    /// `0-3,8,10-11` as the CPU numbers it lists; nothing if it is not such a list.
    pub(super) fn parse_list(text: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        for part in text.trim().split(',') {
            let (first, last) = part.split_once('-').unwrap_or((part, part));
            let (Ok(first), Ok(last)) = (first.trim().parse::<usize>(), last.trim().parse::<usize>()) else {
                return Vec::new();
            };
            if last < first || last - first > 4096 {
                return Vec::new();
            }
            cpus.extend(first..=last);
        }
        cpus
    }

    /// The logical CPUs the process may run on.
    fn allowed() -> Vec<usize> {
        // SAFETY: an all-zero `cpu_set_t` is a valid empty set, and `sched_getaffinity` writes at
        // most the size given.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return Vec::new();
            }
            (0..libc::CPU_SETSIZE as usize).filter(|&cpu| libc::CPU_ISSET(cpu, &set)).collect()
        }
    }

    pub(super) fn physical_cores() -> Option<usize> {
        let siblings = siblings();
        let mut cores = Vec::new();
        for cpu in allowed() {
            // The lowest CPU number of a core names it.
            let core = *siblings.get(cpu)?.iter().min()?;
            if !cores.contains(&core) {
                cores.push(core);
            }
        }
        Some(cores.len()).filter(|&n| n > 0)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn lists_parse() {
            assert_eq!(parse_list("0,16\n"), [0, 16]);
            assert_eq!(parse_list("0-3,8,10-11"), [0, 1, 2, 3, 8, 10, 11]);
            assert_eq!(parse_list("5"), [5]);
            assert!(parse_list("").is_empty());
            assert!(parse_list("3-1").is_empty());
            assert!(parse_list("a-b").is_empty());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_queries_agree() {
        for a in 0..64 {
            assert!(!shares_core(a, a));
            for b in 0..64 {
                assert_eq!(shares_core(a, b), shares_core(b, a), "cpus {a} and {b}");
            }
        }
        if let Some(cores) = physical_cores() {
            assert!(cores >= 1);
        }
    }
}
