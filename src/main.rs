use clap::Parser;
use rayon::prelude::*;

/// store a 5-letter word as a u64, where the 5 least significant bytes are the ascii for the
/// letters of the word
fn word_to_u64(word: &[u8]) -> u64 {
    word.iter()
        .copied()
        .inspect(|&c| {
            if !(b'a' <= c || c <= b'z') {
                panic!(
                    "tried to compress invalid character {} ({:x}) in word {}",
                    c as char,
                    c,
                    std::str::from_utf8(word).unwrap()
                );
            }
        })
        .fold(0, |acc, b| (acc << 8) + b as u64)
}

/// retrieves the word stored by `word_to_u64`
fn u64_to_word(word: u64) -> String {
    std::str::from_utf8(&word.to_be_bytes()[3..])
        .unwrap()
        .to_string()
}

/// load the words out of a word list file with the given `contents`
fn load_words(contents: &str) -> Vec<u64> {
    contents
        .lines()
        .map(|line| {
            assert_eq!(line.len(), 5);
            word_to_u64(line.as_bytes())
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Color {
    Black,
    Yellow,
    Green,
}

#[inline(always)]
fn byte_at_idx(word: u64, idx: usize) -> u8 {
    let shift = 4 - idx;
    ((word >> (shift * 8)) & 0xff) as u8
}

type Response = [Color; 5];

fn compute_response(guess: u64, ans: u64) -> Response {
    let mut counts = [0u8; 26];
    for idx in 0..5 {
        let count_idx = byte_at_idx(ans, idx) - b'a';
        unsafe { *counts.get_unchecked_mut(count_idx as usize) += 1 };
    }
    let mut response = [Color::Black; 5];
    // assign greens
    for (idx, color) in response.iter_mut().enumerate() {
        let guess_char = byte_at_idx(guess, idx);
        let ans_char = byte_at_idx(ans, idx);
        let remaining = unsafe { counts.get_unchecked_mut((guess_char - b'a') as usize) };
        if guess_char == ans_char {
            *remaining -= 1;
            *color = Color::Green;
        }
    }
    // assign yellows
    for (idx, color) in response.iter_mut().enumerate() {
        let guess_char = byte_at_idx(guess, idx);
        let remaining = unsafe { counts.get_unchecked_mut((guess_char - b'a') as usize) };
        if *remaining != 0 {
            *remaining -= 1;
            *color = std::cmp::max(Color::Yellow, *color);
        }
    }
    response
}

/// read response out of the standard input
///
/// Response should be 5 letters, with the meanings:
/// - g for green
/// - y for yellow
/// - x for black
fn read_response() -> Response {
    'outer: loop {
        let mut line = String::new();
        std::io::stdin()
            .read_line(&mut line)
            .expect("reached end of stdin while awaiting response");
        let line = line.trim();
        if line.len() != 5 {
            eprintln!("response must contain exactly 5 characters");
            continue;
        }
        let mut response = [Color::Black; 5];
        for (idx, ch) in line.chars().enumerate() {
            match ch {
                'x' => {} // nothing to do
                'y' => response[idx] = Color::Yellow,
                'g' => response[idx] = Color::Green,
                _ => {
                    eprintln!(
                        "response should be 'g' for green, 'y' for yellow, and 'x' for black"
                    );
                    continue 'outer;
                }
            }
        }
        return response;
    }
}

#[inline]
/// Returns whether `candidate` would be eliminated by the given `guess` and `response`
fn eliminates(guess: u64, response: Response, candidate: u64) -> bool {
    // it would be nice to just have this be compute_response(guess, candidate) == response
    // but the short circuiting in this implementation saves considerable time
    let mut counts = [0u8; 26];
    for idx in 0..5 {
        let count_idx = byte_at_idx(candidate, idx) - b'a';
        unsafe { *counts.get_unchecked_mut(count_idx as usize) += 1 };
    }
    // mark greens
    for (idx, color) in response.iter().copied().enumerate() {
        let guess_char = byte_at_idx(guess, idx);
        let cand_char = byte_at_idx(candidate, idx);
        let remaining = unsafe { counts.get_unchecked_mut((guess_char - b'a') as usize) };
        match color {
            Color::Green => {
                if *remaining == 0 {
                    return true;
                }
                if guess_char != cand_char {
                    return true;
                }
                *remaining -= 1;
            }
            _ => continue,
        }
    }
    // mark yellows
    for (idx, color) in response.iter().copied().enumerate() {
        let guess_char = byte_at_idx(guess, idx);
        let cand_char = byte_at_idx(candidate, idx);
        let remaining = unsafe { counts.get_unchecked_mut((guess_char - b'a') as usize) };
        match color {
            Color::Green => continue,
            Color::Yellow => {
                if *remaining == 0 {
                    return true;
                }
                if guess_char == cand_char {
                    return true;
                }
                *remaining -= 1;
            }
            Color::Black => {
                if *remaining != 0 {
                    return true;
                }
            }
        }
    }
    false
}

#[derive(Debug, Clone)]
struct Solver {
    guesses: Vec<u64>,
    answers: Vec<u64>,
    responses: Vec<Response>,
    hard_mode: bool,
}

impl Solver {
    fn new() -> Self {
        let mut guesses = load_words(include_str!("../data/guess_only.txt"));
        let answers = load_words(include_str!("../data/answers.txt"));
        // put words in `answers` last, meaning they are preferred by max_by_key
        guesses.extend(answers.iter());

        Solver {
            guesses,
            answers,
            responses: Vec::new(),
            hard_mode: false,
        }
    }

    fn make_guess(&self) -> u64 {
        self.guesses
            .par_iter()
            .copied()
            .max_by_key(|g| self.eliminated_words(*g))
            .expect("no more remaining valid guesses =(")
    }

    #[inline]
    /// returns number of eliminations that would occur if `guess` got `response`
    fn count_eliminations(&self, guess: u64, response: Response) -> usize {
        self.answers
            .iter()
            .copied()
            .filter(|cand| eliminates(guess, response, *cand))
            .count()
    }

    /// returns sum of number of words that would be eliminated by `guess` for each remaining
    /// possible answer
    fn eliminated_words(&self, guess: u64) -> usize {
        self.answers
            .iter()
            .copied()
            .map(|ans| {
                let response = compute_response(guess, ans);
                self.count_eliminations(guess, response)
            })
            .sum()
    }

    /// learn from a guess / response pair
    fn learn(&mut self, guess: u64, response: Response) {
        self.answers
            .retain(|ans| !eliminates(guess, response, *ans));
        self.responses.push(response);
        if self.hard_mode {
            self.guesses
                .retain(|ans| !eliminates(guess, response, *ans));
        }
    }
}

fn print_time(duration: std::time::Duration) -> String {
    if duration.as_secs() > 2 {
        return format!("{} s", duration.as_secs());
    }
    if duration.as_millis() > 2 {
        return format!("{} ms", duration.as_millis());
    }
    return format!("{} μs", duration.as_micros());
}

fn main() {
    let mut opts = Options::parse();

    let load_start = std::time::Instant::now();
    let mut solver = Solver::new();
    let load_end = std::time::Instant::now();
    println!(
        "loaded dictionaries in {}",
        print_time(load_end - load_start)
    );
    if opts.hard_mode {
        solver.hard_mode = true;
    }

    // check guesses
    for guess in opts.guesses.iter_mut() {
        *guess = guess.to_lowercase();
        if !guess.chars().all(|c| c.is_ascii_alphabetic()) {
            panic!(
                "guess {} must be only have ASCII alphabetic characters",
                guess
            );
        }
        if guess.len() != 5 {
            panic!(
                "guesses must contain exactly 5 characters, but {} had {}",
                guess,
                guess.len()
            );
        }
    }

    // handle initial guesses
    for guess in opts.guesses {
        println!("what was the response to {}?", guess);
        let response = read_response();
        let guess = word_to_u64(guess.as_bytes());
        solver.learn(guess, response)
    }

    loop {
        // generate guess
        let start = std::time::Instant::now();
        let guess = solver.make_guess();
        let end = std::time::Instant::now();
        let expected_eliminations =
            solver.eliminated_words(guess) as f64 / solver.answers.len() as f64;
        let expected_remnants = solver.answers.len() as f64 - expected_eliminations;
        println!(
            "guess: {} (generated in {}, expected remnants = {:.1})",
            u64_to_word(guess),
            print_time(end - start),
            expected_remnants
        );

        // read and learn from response
        let response = read_response();
        solver.learn(guess, response);

        // print out example remaining words
        let examples: Vec<String> = solver
            .answers
            .iter()
            .take(5)
            .map(|a| u64_to_word(*a))
            .collect();
        println!("example remaining words: {}", examples.join(", "));

        // exit if we are done!
        if response.iter().all(|c| *c == Color::Green) {
            println!("Solved in {} guesses", solver.responses.len());
            break;
        }
    }
}

#[derive(clap::Parser)]
#[clap(name = "wordle")]
struct Options {
    #[clap(short, long)]
    /// if set, all guesses will meet all criteria so far
    hard_mode: bool,

    #[clap(short, long)]
    /// initial guesses (repeat once per guess)
    ///
    /// `wordle` will prompt for the responses to each
    guesses: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_response() {
        use Color::*;
        let answer = word_to_u64(b"abcde");
        assert_eq!(
            compute_response(word_to_u64(b"aabxx"), answer),
            [Green, Black, Yellow, Black, Black]
        );

        assert_eq!(
            compute_response(word_to_u64(b"aaaaa"), answer),
            [Green, Black, Black, Black, Black]
        );

        assert_eq!(
            compute_response(word_to_u64(b"bbbbb"), answer),
            [Black, Green, Black, Black, Black]
        );

        assert_eq!(
            compute_response(word_to_u64(b"edcba"), answer),
            [Yellow, Yellow, Green, Yellow, Yellow]
        );

        assert_eq!(compute_response(word_to_u64(b"vwxyz"), answer), [Black; 5]);
    }

    #[test]
    fn test_eliminates() {
        use Color::*;
        let answer = word_to_u64(b"abcde");
        let guess = word_to_u64(b"aabxx");
        let response = [Green, Black, Yellow, Black, Black];

        // the correct answer should not be eliminated
        assert!(!eliminates(guess, response, answer));

        // yellow where green should be
        assert!(eliminates(guess, response, word_to_u64(b"bzcde")));

        // unknown where green should be
        assert!(eliminates(guess, response, word_to_u64(b"zbcde")));

        // missing yellow
        assert!(eliminates(guess, response, word_to_u64(b"azcde")));

        // yellow in same spot
        assert!(eliminates(guess, response, word_to_u64(b"azbde")));

        // contains black
        assert!(eliminates(guess, response, word_to_u64(b"axcdb")));
    }

    #[test]
    fn regression_test_compute_response() {
        use Color::*;
        // assert_eq!(
        //     compute_response(word_to_u64(b"worry"), word_to_u64(b"purge")),
        //     [Black, Black, Green, Black, Black]
        // );

        // assert_eq!(
        //     compute_response(word_to_u64(b"roate"), word_to_u64(b"purge")),
        //     [Yellow, Black, Black, Black, Green]
        // );

        // assert_eq!(
        //     compute_response(word_to_u64(b"roate"), word_to_u64(b"sling")),
        //     [Black, Black, Black, Black, Black]
        // );

        assert_eq!(
            compute_response(word_to_u64(b"wreck"), word_to_u64(b"crick")),
            [Black, Green, Black, Green, Green]
        );
    }

    #[test]
    fn regression_test_eliminates() {
        use Color::*;
        assert!(eliminates(
            word_to_u64(b"roate"),
            [Black, Black, Black, Black, Green],
            word_to_u64(b"sling")
        ));

        assert!(!eliminates(
            word_to_u64(b"roate"),
            [Black, Black, Yellow, Green, Green],
            word_to_u64(b"paste")
        ));

        assert!(!eliminates(
            word_to_u64(b"crepe"),
            [Black, Black, Black, Yellow, Green],
            word_to_u64(b"paste")
        ));
    }
}
