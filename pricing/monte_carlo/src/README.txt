### Install rust and cargo
$ curl https://sh.rustup.rs -sSf | sh

### Compile
$ cargo build --release

### Run (Homework)
$ cargo run --release 
MC 3.663834775773796
AT: 3.663132601645145

### Run tests
## Convergence comparison
$ cargo test --release -- test::convergence_comparison --nocapture
## Barrier = Inf case, validation with BS formula.
$ cargo test --release -- test::without_barrier --nocapture
MC: 10.442952013033766
AT: 10.44667586207907
BS: 10.4506
## Scan for various S0
$ cargo test --release -- test::s0_scan --nocapture
(see scan.csv. it took 3 minutes in 56 core cpu)
## Measure simulation speed
$ cargo test --release -- test::speed_test --nocapture
Testing simulation speed
number of simulations: 10000000
Monte Carlo: 873.107428ms
Antithetic: 1.167947912s
## Check if random samples are correctly generated
$ cargo test --release -- test::test_random_two_samples --nocapture
test result: ok. (Test passes when n > 1_000_000)