use rayon::prelude::*;

/// store a 5-letter word as a u64, where the 5 least significant bytes are the ascii for the
/// letters of the word
fn word_to_u64(word: &[u8]) -> u64 {
    word.iter().copied().fold(0, |acc, b| (acc << 8) + b as u64)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Black,
    Yellow,
    Green,
}

type Response = [Color; 5];

fn compute_response(guess: u64, ans: u64) -> Response {
    let mut response = [Color::Black; 5];
    for (idx, color) in response.iter_mut().enumerate() {
        if matches_byte_at_index(guess, ans, idx) {
            *color = Color::Green;
        } else if has_byte_at_index(guess, ans, idx) {
            *color = Color::Yellow;
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
/// returns whether the `idx` letter in the word represented by `a` and `b` matches
fn matches_byte_at_index(a: u64, b: u64, idx: usize) -> bool {
    // shift is opposite of index because of endianness
    let shift = 4 - idx;
    (a >> (shift * 8)) & 0xff == (b >> (shift * 8)) & 0xff
}

#[inline]
/// Finds whether `haystack` contains the letter at `idx` in the word represented by `needle`
fn has_byte_at_index(needle: u64, haystack: u64, idx: usize) -> bool {
    // shift is opposite of index because of endianness
    let shift = 4 - idx;
    let byte = (needle >> (shift * 8)) & 0xff;
    for cand_shift in 0..5 {
        if (haystack >> (cand_shift * 8)) & 0xff == byte {
            return true;
        }
    }
    false
}

#[inline]
/// Returns whether `candidate` would be eliminated by the given `guess` and `response`
fn eliminates(guess: u64, response: Response, candidate: u64) -> bool {
    for (idx, color) in response.iter().copied().enumerate() {
        match color {
            Color::Green => {
                if !matches_byte_at_index(guess, candidate, idx) {
                    return true;
                }
            }
            Color::Yellow => {
                if !has_byte_at_index(guess, candidate, idx)
                    || matches_byte_at_index(guess, candidate, idx)
                {
                    return true;
                }
            }
            Color::Black => {
                if has_byte_at_index(guess, candidate, idx) {
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
        }
    }

    fn make_guess(&self) -> u64 {
        if self.answers.is_empty() {
            panic!("no valid words left!");
        }
        if self.answers.len() == 1 {
            // need special case, otherwise it gets confused by the fact that everything eliminates
            // everything
            return self.answers[0];
        }
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
    let load_start = std::time::Instant::now();
    let mut solver = Solver::new();
    let load_end = std::time::Instant::now();
    println!(
        "loaded dictionaries in {}",
        print_time(load_end - load_start)
    );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_response() {
        use Color::*;
        let answer = word_to_u64(b"abcde");
        assert_eq!(
            compute_response(word_to_u64(b"aabxx"), answer),
            [Green, Yellow, Yellow, Black, Black]
        );

        assert_eq!(
            compute_response(word_to_u64(b"aaaaa"), answer),
            [Green, Yellow, Yellow, Yellow, Yellow]
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
        let response = [Green, Yellow, Yellow, Black, Black];

        // the correct answer should not be eliminated
        assert!(!eliminates(guess, response, answer));

        // yellow where green should be
        assert!(eliminates(guess, response, word_to_u64(b"bbcde")));

        // unknown where green should be
        assert!(eliminates(guess, response, word_to_u64(b"zbcde")));

        // missing yellow
        assert!(eliminates(guess, response, word_to_u64(b"azcde")));

        // yellow in same spot
        assert!(eliminates(guess, response, word_to_u64(b"abbde")));

        // contains black
        assert!(eliminates(guess, response, word_to_u64(b"axcdb")));
    }

    #[test]
    fn regression_test_eliminates() {
        use Color::*;
        assert!(eliminates(
            word_to_u64(b"roate"),
            [Black, Black, Black, Black, Green],
            word_to_u64(b"sling")
        ));
    }
}
