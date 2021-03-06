use clap::Parser;
use float_ord::FloatOrd;
use rayon::prelude::*;
use std::collections::HashMap;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        if *color == Color::Green {
            continue;
        }
        let guess_char = byte_at_idx(guess, idx);
        let remaining = unsafe { counts.get_unchecked_mut((guess_char - b'a') as usize) };
        if *remaining != 0 {
            *remaining -= 1;
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
/// optimized to go fast
///
/// Like, really fast.
///
/// This is a column-major table of info about the remaining words
struct WordTable {
    /// these five columns contain the ascii byte in each position of the word
    letter: [Vec<u8>; 5],
    /// these 26 columns contain the number of times a..z appear
    count: [Vec<u8>; 26],
}

impl WordTable {
    #[allow(clippy::needless_range_loop)]
    fn from_words(words: &[u64]) -> Self {
        // kinda awkward to initialize because vec isn't copy
        let mut letter = [(); 5].map(|_| vec![0; words.len()]);
        let mut count = [(); 26].map(|_| vec![0; words.len()]);

        for (word_idx, word) in words.iter().copied().enumerate() {
            for let_idx in 0..5 {
                let byte = byte_at_idx(word, let_idx);
                letter[let_idx][word_idx] = byte;
                count[(byte - b'a') as usize][word_idx] += 1;
            }
        }

        Self { letter, count }
    }

    fn len(&self) -> usize {
        self.letter[0].len()
    }

    fn count_eliminations(&self, guess: u64, response: Response) -> usize {
        // counts of present letters
        let mut present_counts = [0; 26];
        // counts of absent letters
        let mut absent_counts = [0; 26];

        // whether to eliminate word
        let mut eliminate = vec![false; self.len()];

        for (guess_idx, color) in (0..5).zip(response.into_iter()) {
            let guess_byte = byte_at_idx(guess, guess_idx);
            let counts_idx = (guess_byte - b'a') as usize;
            assert!(counts_idx < 26);
            match color {
                Color::Green => {
                    *present_counts.get_mut(counts_idx).unwrap() += 1;
                    assert_eq!(eliminate.len(), self.letter[guess_idx].len());
                    for (eliminate, word_byte) in
                        eliminate.iter_mut().zip(self.letter[guess_idx].iter())
                    {
                        *eliminate |= guess_byte != *word_byte;
                    }
                }
                Color::Yellow => {
                    *present_counts.get_mut(counts_idx).unwrap() += 1;
                    for (eliminate, word_byte) in
                        eliminate.iter_mut().zip(self.letter[guess_idx].iter())
                    {
                        *eliminate |= guess_byte == *word_byte;
                    }
                }
                Color::Black => *absent_counts.get_mut(counts_idx).unwrap() += 1,
            }
        }

        for (counts_idx, (pres_count, abs_count)) in present_counts
            .into_iter()
            .zip(absent_counts.into_iter())
            .enumerate()
        {
            if abs_count > 0 {
                assert_eq!(eliminate.len(), self.count[counts_idx].len());
                for (eliminate, word_count) in
                    eliminate.iter_mut().zip(self.count[counts_idx].iter())
                {
                    *eliminate |= pres_count != *word_count;
                }
            } else if pres_count > 0 {
                assert_eq!(eliminate.len(), self.count[counts_idx].len());
                for (eliminate, word_count) in
                    eliminate.iter_mut().zip(self.count[counts_idx].iter())
                {
                    *eliminate |= *word_count < pres_count;
                }
            }
        }

        eliminate.into_iter().filter(|e| *e).count()
    }
}

#[derive(Debug, Clone)]
struct GuessInfo {
    guess: u64,
    score: f64,
}

#[derive(Debug, Clone)]
struct Solver {
    /// guess-only list
    guesses: Vec<u64>,
    /// answer-only list
    answers: Vec<u64>,
    /// list of entered responses so far
    responses: Vec<Response>,
    /// whether we are playing in hard mode
    hard_mode: bool,
    /// strategy for solver
    strategy: Strategy,
}

impl Solver {
    fn new() -> Self {
        let guesses = load_words(include_str!("../data/guess_only.txt"));
        let answers = load_words(include_str!("../data/answers.txt"));

        Solver {
            guesses,
            answers,
            responses: Vec::new(),
            hard_mode: false,
            strategy: Strategy::Mean,
        }
    }

    fn make_guess(&self) -> GuessInfo {
        let table = WordTable::from_words(&self.answers);
        let guess = self
            .guesses
            .par_iter()
            .copied()
            .chain(self.answers.par_iter().copied())
            .max_by_key(|g| self.score_from_table(*g, &table))
            .expect("no more remaining valid guesses =(");
        let score = self.score_from_table(guess, &table).0;
        GuessInfo { guess, score }
    }

    /// returns sum of number of words that would be eliminated by `guess` for each remaining
    /// possible answer, but faster
    fn score_from_table(&self, guess: u64, table: &WordTable) -> FloatOrd<f64> {
        // cache of response -> words eliminated
        let mut cache = HashMap::new();
        // number of times response has been encountered
        let mut counts = HashMap::new();
        for &ans in &self.answers {
            let response = compute_response(guess, ans);
            cache
                .entry(response)
                .or_insert_with(|| table.count_eliminations(guess, response));
            *counts.entry(response).or_default() += 1usize;
        }
        match self.strategy {
            Strategy::Mean => {
                let total_elims: usize = cache
                    .into_iter()
                    .map(|(resp, elims)| counts.get(&resp).unwrap() * elims)
                    .sum();
                FloatOrd(total_elims as f64 / self.answers.len() as f64)
            }
            Strategy::Median => {
                let mut responses: Vec<Response> = counts.keys().cloned().collect();
                responses.sort_unstable_by_key(|r| cache.get(r).unwrap());
                let mut sum = 0;
                for response in responses {
                    sum += counts.get(&response).unwrap();
                    if sum >= self.answers.len() / 2 {
                        return FloatOrd(*cache.get(&response).unwrap() as f64);
                    }
                }
                unreachable!()
            }
            Strategy::Worst => FloatOrd(cache.values().copied().min().unwrap_or_default() as f64),
        }
    }

    /// learn from a guess / response pair
    fn learn(&mut self, guess: u64, response: Response) {
        if self.hard_mode {
            // only guesses which meet the criteria are valid now
            self.guesses
                .retain(|ans| !eliminates(guess, response, *ans));
        } else {
            // we prefer clues in the answers-only list, so move newly-invalidated answers to the
            // guess-only list
            self.guesses.extend(
                self.answers
                    .iter()
                    .copied()
                    .filter(|ans| eliminates(guess, response, *ans)),
            );
            // we'll delete these from `answers` in a second.
        }
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
    return format!("{} ??s", duration.as_micros());
}

fn format_score(guess_info: &GuessInfo, solver: &Solver) -> String {
    match solver.strategy {
        Strategy::Mean => {
            let expected_eliminations = guess_info.score;
            let expected_remnants = solver.answers.len() as f64 - expected_eliminations;
            format!("expected remnants = {:.1}", expected_remnants)
        }
        Strategy::Median => {
            let expected_eliminations = guess_info.score;
            let expected_remnants = solver.answers.len() as f64 - expected_eliminations;
            format!("median remnants = {:.1}", expected_remnants)
        }
        Strategy::Worst => {
            let expected_eliminations = guess_info.score;
            let expected_remnants = solver.answers.len() as f64 - expected_eliminations;
            format!("worst-case remnants = {:.1}", expected_remnants)
        }
    }
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
    solver.strategy = opts.strategy;

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
        if solver.answers.is_empty() {
            panic!("no more remaining valid answers =(")
        }

        // print out remaining word info
        let examples: Vec<String> = solver
            .answers
            .iter()
            .take(5)
            .map(|a| u64_to_word(*a))
            .collect();
        if solver.answers.len() > 5 {
            println!(
                "{} remaining words. Examples: {}...",
                solver.answers.len(),
                examples.join(", "),
            );
        } else {
            println!(
                "{} remaining words: {}",
                solver.answers.len(),
                examples.join(", "),
            );
        }

        // generate guess
        let start = std::time::Instant::now();
        let guess_info = solver.make_guess();
        let end = std::time::Instant::now();

        println!(
            "guess: {} (generated in {}, {})",
            u64_to_word(guess_info.guess),
            print_time(end - start),
            format_score(&guess_info, &solver)
        );

        // read and learn from response
        let response = read_response();
        solver.learn(guess_info.guess, response);

        // exit if we are done!
        if response.iter().all(|c| *c == Color::Green) {
            println!("Solved in {} guesses", solver.responses.len());
            break;
        }
    }
}

#[derive(clap::ArgEnum, Debug, Clone, Copy)]
enum Strategy {
    /// guess the word with the best mean eliminations
    Mean,
    /// guess the word with the best median eliminations
    Median,
    /// guess the word with the best worst-case eliminations
    Worst,
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

    #[clap(short, long, arg_enum, default_value = "mean")]
    /// which strategy to use
    strategy: Strategy,
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
        assert_eq!(
            compute_response(word_to_u64(b"worry"), word_to_u64(b"purge")),
            [Black, Black, Green, Black, Black]
        );

        assert_eq!(
            compute_response(word_to_u64(b"roate"), word_to_u64(b"purge")),
            [Yellow, Black, Black, Black, Green]
        );

        assert_eq!(
            compute_response(word_to_u64(b"roate"), word_to_u64(b"sling")),
            [Black, Black, Black, Black, Black]
        );

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

    #[test]
    fn test_solver_prefers_answers() {
        // example taken from observed bug
        let crick = word_to_u64(b"crick");
        let crimp = word_to_u64(b"crimp");
        let wreck = word_to_u64(b"wreck");

        let solver = Solver {
            guesses: vec![wreck, crick, crimp],
            answers: vec![crick, crimp],
            responses: vec![],
            hard_mode: false,
            strategy: Strategy::Mean,
        };

        assert_eq!(solver.make_guess().guess, crimp);
    }

    #[test]
    fn test_self_elimination() {
        let guess = word_to_u64(b"creme");
        for word in ["creep", "crepe", "crimp", "crisp", "gripe"] {
            let ans = word_to_u64(word.as_bytes());
            let response = compute_response(guess, ans);
            assert!(
                !eliminates(guess, response, ans),
                "{} eliminated itself",
                word
            );
        }
    }
}
