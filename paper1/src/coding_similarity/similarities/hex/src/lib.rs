extern crate generic_levenshtein;
extern crate ordered_float;

use ordered_float::OrderedFloat;

pub type Score = u32;
pub type MetricCoding = Vec<Score>;
pub type JobCoding = Vec<MetricCoding>;
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

pub mod lev {
    use super::*;
    pub fn compute_similarity_levenshtein_1d(coding_1: &MetricCoding, coding_2: &MetricCoding) -> Similarity {
        let d = generic_levenshtein::distance(coding_1, coding_2);
        let s: Similarity = (1 as Similarity) - (d as Similarity) / (std::cmp::max(coding_1.len(), coding_2.len()) as Similarity);
        s
    }

    //// with consideration of I/O intensive metrics
    //pub fn similarity(coding_1: &JobCoding, coding_2: &JobCoding) -> Similarity {
    //   // Find segments with that contain only zeros
    //   let is_zero_coding: Vec<u32> = coding_1.iter().zip(coding_2)
    //       .map(|(mc_1, mc_2)| 
    //            {
    //                if (mc_1.iter().sum::<u32>() + mc_2.iter().sum::<u32>()) > 0 {1} else {0}
    //            })
    //   .collect();
    //   let n_metrics: u32 = is_zero_coding.iter().sum();

    //   let sim_sum = coding_1.iter().zip(coding_2).zip(is_zero_coding)
    //       .map(|((mc_1, mc_2), are_zero_codings)| {compute_similarity_levenshtein_1d(&mc_1, &mc_2) * (are_zero_codings as f32) })
    //       .sum::<Similarity>();
    //   sim_sum /(n_metrics as Similarity)
    //}

    // Naive implementation
    pub fn similarity(coding_1: &JobCoding, coding_2: &JobCoding) -> Similarity {
       let n_metrics = std::cmp::max(coding_1.len(), coding_2.len());
       let sim_sum = coding_1.iter().zip(coding_2)
           .map(|(mc_1, mc_2)| {
               compute_similarity_levenshtein_1d(&mc_1, &mc_2)
           })
           .sum::<Similarity>();
       sim_sum / (n_metrics as Similarity)
    }
}



pub mod native {
    use super::*;
    // HEX_NATIVE Helper
    pub fn compute_similarity_sliding_windows_1d(coding_1: &Vec<Score>, coding_2: &Vec<Score>) -> Similarity {
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


    pub fn similarity(coding_1: &JobCoding, coding_2: &JobCoding) -> Similarity {
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

        // Tupel ((coding1, coding2), areZeroJobCodings)
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
        .sum::<Similarity>();
        sim_sum /(n_metrics as Similarity)
    }
}


pub mod quant {
    use super::*;
    pub mod phase_combis_2;

    use phase_combis_2::PhaseNum;
    use phase_combis_2::PhaseLength;

    //pub type Score = u32;
    pub type Phase = MetricCoding;
    pub type PhaseSet = Vec<Phase>;
    //pub type MetricPhaseCoding = Vec<PhaseCoding>;
    //pub type MetricCoding = Vec<PhaseCoding>;
    //pub type JobCoding = Vec<MetricCoding>;
    //pub type Similarity = f32;
    pub type QSum= f32;


    fn fmax<T: PartialOrd>(a: T, b: T) -> std::cmp::Ordering {
        if a < b {
            std::cmp::Ordering::Less
        }
        else if a > b {
            std::cmp::Ordering::Greater
        }
        else {
            std::cmp::Ordering::Equal
        }
    }

    /// Sum up quotients (low value / high value)
    pub fn sum_quotients(coding_1:&[Score], coding_2:&[Score]) -> Similarity {
        assert!(coding_1.len() == coding_2.len());
        let l = coding_1.len();
        let mut quotients: Vec<Similarity> = vec![0.0; l];

        for i in 0..l {
            quotients[i] = if coding_1[i] < coding_2[i] {
                coding_1[i] as Similarity / coding_2[i] as Similarity
            }
            else {
                coding_2[i] as Similarity / coding_1[i] as Similarity
            };
        }
        let sum: Similarity = quotients.iter().sum();
        sum
    }


    /// Get max quotient and max length of two non-equal sized phases
    /// given 222282 2293
    /// Test all possibilites 
    /// 229300 222282
    /// 022930 222282
    /// 002293 222282
    /// return (quotient_sum, length)
    pub fn get_best_phase_match(coding_1: &Vec<Score>, coding_2: &Vec<Score>) -> (Similarity, PhaseLength) {
        let c1: &Vec<Score>;    
        let c2: &Vec<Score>;
        if coding_1.len() > coding_2.len() {
            c1 = coding_2;
            c2 = coding_1;
        }
        else {
            c1 = coding_1;
            c2 = coding_2;
        }

        let l1 = c1.len();
        let l2 = c2.len();
        let max_len = l2;

        let mut quotient_sums: Vec<Similarity> = Vec::new();
        for shift in 0..l2-l1+1 {
            quotient_sums.push(sum_quotients(&c1[..], &c2[shift..(shift+l1)]));
        }

        let max_value = quotient_sums.iter().max_by(|x, y| fmax(**x, **y));

        match max_value {
            Some(&m) => (m, max_len as PhaseNum),
            None => (0.0, 0),
        }
    }



    // get quotient sums
    // 
    pub fn get_job_qsums_1d(phases_1: &PhaseSet, phases_2: &PhaseSet) -> Vec<(QSum, PhaseLength)> {
        let ps1: &PhaseSet;    
        let ps2: &PhaseSet;
        if phases_1.len() > phases_2.len() {
            ps1 = phases_2;
            ps2 = phases_1;
        }
        else {
            ps1 = phases_1;
            ps2 = phases_2;
        }

        // Constraint l2 >= l1
        let l1 = ps1.len();
        let l2 = ps2.len();

        let mut qsums: Vec<(QSum, PhaseLength)> = Vec::new();
        if l1 == 0 {
            for p2 in ps2 {
                qsums.push((0.0, (p2.len() as PhaseLength)));
            }
            return qsums;
        }

        let combis = phase_combis_2::PhaseCombinations::new(l1 as PhaseNum, l2 as PhaseNum);
        let mut best_qsums: Vec<(QSum, PhaseLength)> = Vec::new();

        let mut max_weighted_mean = 0.0;
        // go through all combinations
        //
        for combi in combis {
            let mut qsums: Vec<(QSum, PhaseLength)> = Vec::new();
            for (idx_ps1, idx_ps2) in combi.to_indices_attached().iter().enumerate() {
                let (qsum, phase_len) = get_best_phase_match(&ps1[idx_ps1], &ps2[*idx_ps2 as usize]);
                qsums.push((qsum, phase_len as PhaseNum));
            }
            // Insert tuple (0.0, phase_len)
            for idx in combi.to_indices_not_attached().iter() {
                let phase_len = ps2[*idx as usize].len();
                let qsum = 0.0;
                qsums.push((qsum, phase_len as PhaseNum));
            }
            //let weighted_mean = weighted_mean(&qsums);
            let weighted_mean = qsums.iter().fold((0.0, 0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
            let weighted_mean = weighted_mean.0 / (weighted_mean.1 as f32);
            if weighted_mean > max_weighted_mean {
                max_weighted_mean = weighted_mean;
                best_qsums = qsums;
            }
        }
        best_qsums
    }


    pub fn sum_phase_len(phases: &Vec<Vec<Score>>) -> usize {
        let mut sum = 0;
        for phase_coding in phases {
            sum += phase_coding.len();
        }
        sum
    }

    /*
       job1_phase1_coding : [[2,2,2,9,3], [9,1,1,1]]
       job1_phase2_coding : [[9,3], [1]]

       job2_phase1_coding : [[1]] 
       job2_phase2_coding : [[2,2,2,2,8,2,2], [1], [8,1,1]] 


       Step 1:
       job1_phase1_coding : [[2,2,2,9,3], [9,1,1,1]]
       job2_phase1_coding : [[-,-,-,-,-], [-,1,-,-]] 
       [(0;6), (1;4)]
       sim = 1/10

       job1_phase2_coding : [[-,-,-,-,9,3,-], [1], [-,-,-]]
       job2_phase2_coding : [[2,2,2,2,8,2,2], [1], [8,1,1]] 
       [(1.55;7), (1;1), (0;3)]
       sim = (1.55 + 1 + 0)/11
       */

    //pub fn job_similarity_2d(metric_codings_1: &Vec<Vec<Vec<Score>>>, metric_codings_2: &Vec<Vec<Vec<Score>>>) -> Similarity {
    pub fn similarity(job_1: &JobCoding, job_2: &JobCoding) -> Similarity {

        let metric_codings_1 = detect_phases_2d(&job_1);
        let metric_codings_2 = detect_phases_2d(&job_2);

        let mut qsumss: Vec<Vec<(QSum, PhaseLength)>> = Vec::new();
        for (metric_coding_1, metric_coding_2) in metric_codings_1.iter().zip(metric_codings_2) {
            qsumss.push(get_job_qsums_1d(&metric_coding_1, &metric_coding_2));
        }
        //println!("{:?}", qsumss);

        // Set of quotient sums
        let (q, l) = qsumss.iter().fold(
            (0.0, 0), |acc, x| {
                let tmp = x.iter().fold(
                    (0.0, 0), |acc2, x2| {
                        (acc2.0 + x2.0, acc2.1 + x2.1)
                    }); 
                (acc.0 + tmp.0, acc.1 + tmp.1)
            });

        q / (l as QSum)
    }



    //pub fn detect_phases_1d(data: &Vec<Score>) -> Vec<Vec<Score>> {
    pub fn detect_phases_1d(data: &MetricCoding) -> Vec<Phase> {
        let data_length = data.len();
        let mut phases: Vec<(usize, usize)> = Vec::new();
        let mut idx = 0;
        while idx < data_length {
            while (idx < data_length) && (data[idx] == (0 as Score)) {
                idx = idx + 1;
            }
            let start = idx;
            while (idx < data_length) && (data[idx] != (0 as Score)) {
                idx = idx + 1;
            }
            let stop = idx;
            phases.push((start, stop));
        }
        if phases[phases.len()-1].0 == data_length {
            phases.pop();
        }
        let mut ps: Vec<Vec<Score>> = Vec::new();
        for phase in phases {
            let mut p:Vec<Score> = Vec::new();
            for i in phase.0..(phase.1) {
                p.push(data[i]);
            }
            ps.push(p);
        }
        ps
    }



    /// Find phases in metrics
    /// returns a nested vector Job<Metric<Phases>>>
    pub fn detect_phases_2d(metric_codings: &Vec<Vec<Score>>) -> Vec<Vec<Vec<Score>>> {
        let mut res = Vec::new(); 
        for metric_coding in metric_codings {
            res.push(detect_phases_1d(metric_coding));
        }
        res
    }
}


#[cfg(test)]
mod tests_levenshtein {
    use assert_approx_eq::assert_approx_eq;
    use super::*;
    use super::lev::*;

    #[test]
    fn test_levenshtein() {
        let c1 = vec![1, 2, 3, 4, 5];
        let c2 = vec![1, 2, 3, 4, 5];
        assert_eq!(generic_levenshtein::distance(&c1, &c2), 0);
    }

    #[test]
    fn test_similarity_levenshtein() {
        let c1 = vec![1, 2, 3, 4];
        let c2 = vec![1, 2, 3, 4, 5];
        assert_approx_eq!(compute_similarity_levenshtein_1d(&c1, &c2), 0.8, 0.001);
    }

    #[test]
    fn test_similarity_equal_codings() {
       let c1 = vec![
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 4, 0, 4, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
       ];
       let c2 = vec![
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 4, 0, 4, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
       ];
       assert_approx_eq!(lev::similarity(&c1, &c2), 1.0, 0.001);
    }


    #[test]
    fn test_similarity_different_codings() {
       let c1 = vec![
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
           vec![0, 4, 0, 4, 0, 0, 1, 0, 0, 1], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
       ];
       let c2 = vec![
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 4, 0, 4, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
           vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
       ];
       assert_approx_eq!(lev::similarity(&c1, &c2), 0.9, 0.001);
    }
}

#[cfg(test)]
mod tests_native {
    use assert_approx_eq::assert_approx_eq;
    use super::*;
    use super::native::*;

    #[test]
    fn test_compute_similarity_sliding_windows_1d() {
        let c1: Vec<Score> = vec![0, 8, 16, 9, 1, 8, 0, 0];
        let c2: Vec<Score> = vec![7, 16, 5, 0, 9];
        assert_approx_eq!(compute_similarity_sliding_windows_1d(&c1, &c2), 0.5703125, 0.001);

        let c1: Vec<Score> = vec![0, 8, 16, 9, 1, 8, 0, 0];
        let c2: Vec<Score> = vec![0, 8, 16, 9, 1, 8, 0, 0];
        assert_approx_eq!(compute_similarity_sliding_windows_1d(&c1, &c2), 1.0, 0.001);
    }

    #[test]
    fn test_similarity() {
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
        assert_approx_eq!(similarity(&c1, &c2), 0.7937, 0.001);
    }

//4296426

    #[test]
    fn test_quant_detect_phases() {
        let c1 = vec![
            vec![4,4,4,4,4,4,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,0,15,4],
            vec![15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,4,15,15],
            vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
            vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,4,0],
            vec![0,0,4,4,0,0,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15],
            vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![2,2,2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
        ];


        let p1 = vec![
            vec![vec![4,4,4,4,4,4,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15], vec![15,4]],
            vec![vec![15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,4,15,15]],
            vec![vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]],
            vec![vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4], vec![4]],
            vec![vec![4,4], vec![15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![vec![2,2,2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]]
        ];

        assert_eq!(quant::detect_phases_2d(&c1), p1);

        let c2 = vec![
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ];

        let p2 = vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![],
        ];
        assert_eq!(quant::detect_phases_2d(&c2), p2);
    }

    #[test]
    fn test_quant_similarity() {
        let c1 = vec![
            vec![4,4,4,4,4,4,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,0,15,4],
            vec![15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,4,15,15],
            vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
            vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,4,0],
            vec![0,0,4,4,0,0,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15],
            vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![2,2,2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
        ];

        let c2 = vec![
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            vec![0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
            vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ];

        assert_approx_eq!(quant::similarity(&c1, &c2), 10.0, 0.001);
    }



    #[test]
    fn test_quant_qsum_metric() {
        let p1 = vec![vec![4,4,4,4,4,4,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15], vec![15,4]];
        let p2 = vec![vec![]];

        let mut qsumss: Vec<Vec<(quant::QSum, quant::phase_combis_2::PhaseLength)>> = Vec::new();
        qsumss.push(quant::get_job_qsums_1d(&p1, &p2));

        for qsums in qsumss {
            println!("{:?}", qsums);
        }
    }


    #[test]
    fn test_quant_qsum() {
        let p1 = vec![
            vec![vec![4,4,4,4,4,4,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15], vec![15,4]],
            vec![vec![15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,4,15,15]],
            vec![vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]],
            vec![vec![4,4,4,4,4,4,15,15,15,4,4,4,4,4,4,4,4,4,4,4,4,4,4], vec![4]],
            vec![vec![4,4], vec![15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![vec![2,2,2,2,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]]
        ];

        let p2 = vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![],
            vec![vec![8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]],
            vec![],
        ];

        let mut qsumss: Vec<Vec<(quant::QSum, quant::phase_combis_2::PhaseLength)>> = Vec::new();
        for (metric_coding_1, metric_coding_2) in p1.iter().zip(p2) {
            qsumss.push(quant::get_job_qsums_1d(&metric_coding_1, &metric_coding_2));
        }

        let expected = vec![
            vec![(0.0, 23), (0.0, 2)],
            vec![(0.0, 26)],
            vec![(0.0, 26)],
            vec![(0.0, 23), (0.0, 1)],
            vec![(0.0, 2), (0.0, 20)],
            vec![(26.0, 30)],
            vec![(0.0, 26)],
            vec![(26.0, 30)],
            vec![(0.0, 20)]
        ];

        assert_eq!(expected, qsumss);
    }


    #[test]
    fn test_quant_qsum_minimal() {
        let m1 = vec![vec![4,4,4,4,4,4,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15], vec![15,4]];
        let m2 = vec![];
        let qsums = quant::get_job_qsums_1d(&m1, &m2);
        assert_eq!(qsums, [(0.0, 23), (0.0, 2)]);
    }


    #[test]
    fn test_quant_get_best_match() {
        let c1 = vec![4,4,4,4,4,4,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15];
        let c2 = vec![];
        let best_match = quant::get_best_phase_match(&c1, &c2);
        assert_eq!((0.0, 23), best_match);

        let c1 = vec![15,4];
        let c2 = vec![];
        let best_match = quant::get_best_phase_match(&c1, &c2);
        assert_eq!((0.0, 2), best_match);
    }
}


