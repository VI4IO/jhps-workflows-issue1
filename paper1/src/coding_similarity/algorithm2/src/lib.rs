extern crate generic_levenshtein;
extern crate ordered_float;

use ordered_float::OrderedFloat;

pub type CodingType = u32;
pub type SimType = f32;


pub fn compute_similarity_1d(coding_1: &Vec<CodingType>, coding_2: &Vec<CodingType>) -> SimType {
    let d = generic_levenshtein::distance(coding_1, coding_2);
    let s: SimType = (1 as SimType) - (d as SimType) / (std::cmp::max(coding_1.len(), coding_2.len()) as SimType);
    s
}


pub fn compute_similarity_sliding_windows_1d(coding_1: &Vec<CodingType>, coding_2: &Vec<CodingType>) -> SimType {
    let c1;
    let c2;

    // c1 is always larger or equal compared to c2
    if coding_1.len() < coding_2.len() {
        c1 = coding_1;
        c2 = coding_2;
    }
    else {
        c2 = coding_1;
        c1 = coding_2;
    }

    let l1 = c1.len();
    let l2 = c2.len();

    let sim:OrderedFloat<f32> = (0..(l2-l1+1)) // (0..1) -> [0], (0..2) -> [0, 1], ...
        .map(|x| {
            let idx1 = x;
            let idx2 = l1 + idx1;
            let sum: f32 = c1.iter().zip(&c2[idx1..idx2]).map( // sum_{i=1}^{LA}
                |(a, b)|  {
                    let diff = if b>a {b-a} else {a-b};
                    ((16 - diff) as f32) * 0.0625 // 1 - Normalized-Difference
                }
                ).sum();
            OrderedFloat(sum)
        })
        .max()
        .unwrap();
    sim.into_inner() / (c2.len() as f32)  // sum / LB
}


pub fn compute_similarity_levenshtein_2d(coding_1: &Vec<Vec<CodingType>>, coding_2: &Vec<Vec<CodingType>>) -> SimType {
    // Find segments with that contain only zeros
    let z: Vec<u32> = coding_1.iter().zip(coding_2)
        .map(|(mc_1, mc_2)| 
            {
                if (mc_1.iter().sum::<u32>() + mc_2.iter().sum::<u32>()) > 0 {1} else {0}
            })
        .collect();
    let n_metrics: u32 = z.iter().sum();

    let sim_sum = coding_1.iter().zip(coding_2).zip(z)
       .map(|((mc_1, mc_2), cz)| { compute_similarity_1d(&mc_1, &mc_2) })
       .sum::<SimType>();
    sim_sum /(n_metrics as SimType)
}


pub fn compute_similarity_2d(coding_1: &Vec<Vec<CodingType>>, coding_2: &Vec<Vec<CodingType>>) -> SimType {
    // Find segments that contain only zeros
    // mc_1, mc_2 : metric codings
    // z: binary vector indicates I/O intensive metrics: 1 -> I/O intensive, 0-> I/O non-intensive
    // n_metrics: number of I/O intesive metrics
    let z: Vec<u32> = coding_1.iter().zip(coding_2)
        .map(|(mc_1, mc_2)| 
            {
                if (mc_1.iter().sum::<u32>() + mc_2.iter().sum::<u32>()) > 0 {1} else {0}
            })
        .collect();
    let n_metrics: u32 = z.iter().sum();

    // Tupel ((coding1, coding2), areZeroCodings)
    // are_zero_codings is just for optimization
    let sim_sum = coding_1.iter().zip(coding_2).zip(z)
       .map(|((mc_1, mc_2), are_zero_codings)| 
            {
                if are_zero_codings != 0 {
                    compute_similarity_sliding_windows_1d(&mc_1, &mc_2)
                } 
                else {
                    0.0
                } 
            })
       .sum::<SimType>();
    sim_sum /(n_metrics as SimType)
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
    fn test_compute_similarity_1d() {
        let c1 = vec![1, 2, 3, 4];
        let c2 = vec![1, 2, 3, 4, 5];
        assert_approx_eq!(compute_similarity_1d(&c1, &c2), 0.8, 0.001);
    }

    #[test]
    fn test_compute_similarity_sliding_windows_1d() {
        let c1: Vec<CodingType> = vec![0, 8, 16, 9, 1, 8, 0, 0];
        let c2: Vec<CodingType> = vec![7, 16, 5, 0, 9];
        assert_approx_eq!(compute_similarity_sliding_windows_1d(&c1, &c2), 0.5703125, 0.001);

        let c1: Vec<CodingType> = vec![0, 8, 16, 9, 1, 8, 0, 0];
        let c2: Vec<CodingType> = vec![0, 8, 16, 9, 1, 8, 0, 0];
        assert_approx_eq!(compute_similarity_sliding_windows_1d(&c1, &c2), 1.0, 0.001);
    }

    #[test]
    fn test_compute_similarity_2d() {
        // job1 
        // metric1: vec![1, 2, 3, 4]
        // metric2: vec![0, 1, 0, 0]
        //
        // job2
        // metric1: vec![1, 2, 3, 4, 5]
        // metric2: vec![0, 0, 0, 0, 1]
        //
        // Similarities:
        // metric1: (1 - 1/5) = 0.8 
        // metric2: (1 - 2/5) = 0.6
        // mean: (0.8 + 0.6) / 2 = 0.7
        
        let c1 = vec![vec![1, 2, 3, 4], vec![0, 1, 0, 0]];
        let c2 = vec![vec![1, 2, 3, 4, 5], vec![0, 0, 0, 0, 1]];
        assert_approx_eq!(compute_similarity_2d(&c1, &c2), 0.7, 0.001);
    }
}




//def compute_similarity(probe_sec:pd.Series, group_pri:pd.DataFrame, eps:float, dist_type:str) -> pd.DataFrame:
//    if dist_type == 'levenshtein':
//        sim: pd.Series = group_pri.apply(compute_dist_levenshtein, args=(probe_sec,), axis=1)
//    elif dist_type == 'hex':
//        sim: pd.Series = group_pri.apply(compute_dist_hex, args=(probe_sec,), axis=1)
//    else:
//        raise SystemExit('Distance type %s is not supported.' % dist_type)
//    #sim.index=['sim']
//    return sim
