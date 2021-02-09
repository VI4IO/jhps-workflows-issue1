extern crate generic_levenshtein;
extern crate ordered_float;

pub type Score = u32;
pub type JobCoding = Vec<Score>;
pub type Similarity = f32;

pub fn convert_to_coding(coding: String) -> Vec<Score> {
    let split = coding.split(":");
    let vec: Vec<Score> = split
        .filter(|s| !s.is_empty())
        //.map(|s| s.parse::<F>().unwrap()) 
        .map(|s| s.parse().unwrap()) 
        .collect();
    vec
}

pub mod all {
    use super::*;
    // BIN similarity
    // HEX_LEV Helper
    pub fn similarity(coding_1: &JobCoding, coding_2: &JobCoding) -> Similarity {
        let d = generic_levenshtein::distance(coding_1, coding_2);
        let s: Similarity = (1 as Similarity) - (d as Similarity) / (std::cmp::max(coding_1.len(), coding_2.len()) as Similarity);
        s
    }
}

pub mod aggzeros {
    use super::*;
    pub fn aggzeros(coding: &JobCoding) -> JobCoding {
        let mut coding_aggzeros: JobCoding = vec![];
        let mut prev_elem: Score = 1;
        for code in coding {
            if (prev_elem == 0) && (*code == 0) {
                // skip
            }
            else {    
                coding_aggzeros.push(*code);
            }
            prev_elem = *code;
        }
        coding_aggzeros
    }
    
    // BIN similarity
    pub fn similarity(coding_1: &JobCoding, coding_2: &JobCoding) -> Similarity {
        let coding_aggzeros_1 = aggzeros(coding_1);
        let coding_aggzeros_2 = aggzeros(coding_2);
        let d = generic_levenshtein::distance(&coding_aggzeros_1, &coding_aggzeros_2);
        let s: Similarity = (1 as Similarity) - (d as Similarity) / (std::cmp::max(coding_aggzeros_1.len(), coding_aggzeros_2.len()) as Similarity);
        s
    }
}


#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;

    #[test]
    fn test_levenshtein() {
        let c1 = vec![1, 2, 3, 4, 5];
        let c2 = vec![1, 2, 3, 4, 5];
        assert_eq!(generic_levenshtein::distance(&c1, &c2), 0);
    }

    #[test]
    fn test_compute_similarity_all() {
        let c1 = vec![1, 2, 3, 4];
        let c2 = vec![1, 2, 3, 4, 5];
        assert_approx_eq!(all::similarity(&c1, &c2), 0.8, 0.001);
    }

    #[test]
    fn test_compute_similarity_aggzeros() {
        let c1 = vec![1, 0, 0, 2, 2];
        let c2 = vec![1, 0, 0, 0, 2, 2];
        assert_approx_eq!(aggzeros::similarity(&c1, &c2), 1.0, 0.001);
    }


    #[test]
    fn test_aggzeros() {
        let c1 = vec![141,143,142,238,132,486,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        let c2 = vec![141,143,142,238,132,486,38,0,128,0];
        assert_eq!(aggzeros::aggzeros(&c1), c2);
    }

    #[test]
    fn test_aggzeros_similarity() {
        let c1 = vec![141,143,142,238,132,486,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        let c2 = vec![239,239,255,255,239,239,511,511,511,511,511,511,511,511,511,511,511,511,511,511,511,511,511,502,511,503];
        assert_approx_eq!(aggzeros::similarity(&c1, &c2), 0.0, 0.001);
    }

}
