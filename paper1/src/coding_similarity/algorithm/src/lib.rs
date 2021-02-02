mod phase_combis_2;


use phase_combis_2::PhaseNum;
use phase_combis_2::PhaseLength;

pub type CodingType = u32;
pub type SimType = f32;
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
pub fn sum_quotients(coding_1:&[CodingType], coding_2:&[CodingType]) -> SimType {
    assert!(coding_1.len() == coding_2.len());
    let l = coding_1.len();
    let mut quotients: Vec<SimType> = vec![0.0; l];

    for i in 0..l {
        quotients[i] = if coding_1[i] < coding_2[i] {
            coding_1[i] as SimType / coding_2[i] as SimType
        }
        else {
            coding_2[i] as SimType / coding_1[i] as SimType
        };
    }
    let sum: SimType = quotients.iter().sum();
    sum
}


/// Get max quotient and max length of two non-equal sized phases
/// given 222282 2293
/// Test all possibilites 
/// 229300 222282
/// 022930 222282
/// 002293 222282
/// return (quotient_sum, length)
fn get_best_phase_match(coding_1: &Vec<CodingType>, coding_2: &Vec<CodingType>) -> (SimType, PhaseLength) {
    let c1: &Vec<CodingType>;    
    let c2: &Vec<CodingType>;
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

    let mut quotient_sums: Vec<SimType> = Vec::new();
    for shift in 0..l2-l1+1 {
        quotient_sums.push(sum_quotients(&c1[..], &c2[shift..(shift+l1)]));
    }

    let max_value = quotient_sums.iter().max_by(|x, y| fmax(**x, **y));

    match max_value {
       Some(&m) => (m, max_len as PhaseNum),
       None => (0.0, 0),
    }
}



///// Helper function for find max value in a f32 vector
//fn max_f32(data: &Vec<SimType>) -> SimType {
//    let mut max: SimType = 0.0;
//    for d in data {
//        if *d > max {
//            max = *d;
//        }
//    }
//    max
//}

///// (sim1*len1 + sim2*len2 + sim3*len4) / (len1 + len2 + len3)
//fn weighted_mean(data: &Vec<(SimType, PhaseLength)>) -> SimType {
//   let mut len_sum: PhaseLength = 0;
//   let mut weighted_sim_sum: SimType = 0.0;
//   for (sim, len) in data {
//       len_sum = len_sum + *len;
//       weighted_sim_sum = weighted_sim_sum + sim / (*len as SimType);
//   }
//   weighted_sim_sum / (len_sum as SimType)
//}


pub fn get_job_qsums_1d(job_coding_1: &Vec<Vec<CodingType>>, job_coding_2: &Vec<Vec<CodingType>>) -> Vec<(QSum, PhaseLength)> {
    let c1: &Vec<Vec<CodingType>>;    
    let c2: &Vec<Vec<CodingType>>;
    if job_coding_1.len() > job_coding_2.len() {
        c1 = job_coding_2;
        c2 = job_coding_1;
    }
    else {
        c1 = job_coding_1;
        c2 = job_coding_2;
    }

    let l1 = c1.len();
    let l2 = c2.len();

    if l1 == 0 {
        return vec![(0.0, (l2 as PhaseLength))];
    }

    let combis = phase_combis_2::PhaseCombinations::new(l1 as PhaseNum, l2 as PhaseNum);
    let mut best_qsums: Vec<(QSum, PhaseLength)> = Vec::new();

    let mut max_weighted_mean = 0.0;
    // go through all combinations
    for combi in combis {
       let mut qsums: Vec<(QSum, PhaseLength)> = Vec::new();
       for (idx_c1, idx_c2) in combi.to_indices_attached().iter().enumerate() {
           let (qsum, phase_len) = get_best_phase_match(&c1[idx_c1], &c2[*idx_c2 as usize]);
           qsums.push((qsum, phase_len));
       }
       for idx in combi.to_indices_not_attached().iter() {
           let phase_len = c2[*idx as usize].len();
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


pub fn sum_phase_len(phases: &Vec<Vec<CodingType>>) -> usize {
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

pub fn job_similarity_2d(metric_codings_1: &Vec<Vec<Vec<CodingType>>>, metric_codings_2: &Vec<Vec<Vec<CodingType>>>) -> SimType {

    let mut qsumss: Vec<Vec<(QSum, PhaseLength)>> = Vec::new();
    for (metric_coding_1, metric_coding_2) in metric_codings_1.iter().zip(metric_codings_2) {
        qsumss.push(get_job_qsums_1d(metric_coding_1, metric_coding_2));
    }
    //println!("{:?}", qsumss);
    
    let (q, l) = qsumss.iter().fold(
        (0.0, 0), |acc, x| {
            let tmp = x.iter().fold(
                (0.0, 0), |acc2, x2| {
                    (acc2.0 + x2.0, acc2.1 + x2.1)
                }); 
            (acc.0 + tmp.0, acc.1 + tmp.1)
        }
    );
   
    q / (l as QSum)
}



pub fn detect_phases_1d(data: &Vec<CodingType>) -> Vec<Vec<CodingType>> {
	let data_length = data.len();
	let mut phases: Vec<(usize, usize)> = Vec::new();
	let mut idx = 0;
	while idx < data_length {
		while (idx < data_length) && (data[idx] == (0 as CodingType)) {
			idx = idx + 1;
		}
		let start = idx;
		while (idx < data_length) && (data[idx] != (0 as CodingType)) {
			idx = idx + 1;
		}
		let stop = idx;
		phases.push((start, stop));
	}
	if phases[phases.len()-1].0 == data_length {
		phases.pop();
	}
    let mut ps: Vec<Vec<CodingType>> = Vec::new();
    for phase in phases {
        let mut p:Vec<CodingType> = Vec::new();
        for i in phase.0..(phase.1) {
            p.push(data[i]);
        }
        ps.push(p);
    }
    ps
}



/// Find phases in metrics
/// returns a nested vector Job<Metric<Phases>>>
pub fn detect_phases_2d(metric_codings: &Vec<Vec<CodingType>>) -> Vec<Vec<Vec<CodingType>>> {
    let mut res = Vec::new(); 
    for metric_coding in metric_codings {
        res.push(detect_phases_1d(metric_coding));
    }
    res
}



#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;

    //#[test]
    //fn test_phase_combinations() {
    //    let res = get_phase_combinations(3, 5);
    //    println!("{:?}", res);
    //    assert_eq!(res[0], Combination{used_idxs: vec![0, 1], unused_idxs: vec![2]});
    //    assert_eq!(res[1], Combination{used_idxs: vec![1, 2], unused_idxs: vec![0]});
    //    assert_eq!(res[2], Combination{used_idxs: vec![0, 2], unused_idxs: vec![1]});
    //}

    //#[test]
    //fn test_print() {
    //    let coding_1 = vec![1, 1, 1, 1];
    //    let coding_2 = vec![2, 1, 1, 1];
    //    //println!("{:?} {:?} {}", coding_1, coding_2, sum_quotients(&coding_1, &coding_2));

    //    let job_coding_1: Vec<Vec<CodingType>> = vec![vec![2, 2, 9, 3], vec![9, 1, 1]];
    //    let job_coding_2: Vec<Vec<CodingType>> = vec![vec![2, 2, 2, 2, 8, 2], vec![1], vec![8, 1, 1]];
    //    //println!("job qsums 1d: {:?}", get_job_qsums_1d(&job_coding_1, &job_coding_2));
    //    assert!(false);
    //}


    #[test]
    fn test_sum_quotients() {
        // 0.5 + 1 + 1 + 1 = 3.5
        let coding_1 = vec![1.0, 1.0, 1.0, 1.0];
        let coding_2 = vec![2.0, 1.0, 1.0, 1.0];
        assert_eq!(sum_quotients(&coding_1, &coding_2), 3.5);
    }

    #[test]
    fn test_sliding_similarity() {
        //let coding_1 = vec![2, 2, 9, 3];
        //let coding_2 = vec![2, 2, 2, 2, 8, 2];

        let coding_1 = vec![2.0, 2.0, 9.0, 3.0];
        let coding_2 = vec![2.0, 2.0, 2.0, 2.0];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 2.889, 0.001);
        let coding_1 = vec![2.0, 2.0, 9.0, 3.0];
        let coding_2 = vec![2.0, 2.0, 2.0, 8.0];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 2.597, 0.001);
        let coding_1 = vec![2.0, 2.0, 9.0, 3.0];
        let coding_2 = vec![2.0, 2.0, 8.0, 2.0];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 3.556, 0.001);

        // 2/2 + 2/2 + 8/9 + 2/3 = 3.5555556
        let coding_1 = vec![2.0, 2.0, 9.0, 3.0];
        let coding_2 = vec![2.0, 2.0, 2.0, 2.0, 8.0, 2.0];
        let (qsum, len) = get_best_phase_match(&coding_1, &coding_2);
        assert_approx_eq!(qsum, 3.556, 0.001);
        assert_eq!(len, 6);

        let coding_1 = vec![8.0];
        let coding_2 = vec![8.0];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 1.0, 0.001);

        let coding_1 = vec![32.0,175.0,128.0,128.0];
        let coding_2 = vec![32.0,175.0,128.0,128.0];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 4.0, 0.001);

        let coding_1 = vec![8.0];
        let coding_2 = vec![8.0];
        let (sim, len) = get_best_phase_match(&coding_1, &coding_2);
        assert_approx_eq!(sim, 1.0/(len as SimType), 0.001);

        let coding_1 = vec![32.0,175.0,128.0,128.0];
        let coding_2 = vec![32.0,175.0,128.0,128.0];
        let (qsum, len) = get_best_phase_match(&coding_1, &coding_2);
        assert_approx_eq!(qsum, 4.0, 0.001);
        assert_eq!(len, 4);
    }


    //#[test]
    ///// (sim1*len1 + sim2*len2 + sim3*len4) / (len1 + len2 + len3)
    //fn test_weighted_mean() {
    //    let mut data: Vec<(SimType, PhaseLength)> = Vec::new();
    //    data.push((1.0, 1));
    //    data.push((1.0, 1));
    //    data.push((1.0, 4));
    //    let sim = weighted_mean(&data);
    //    //println!("sim {}", sim);
    //    assert_approx_eq!(sim, 1.0, 0.001);
    //}

    #[test]
    fn test_get_job_qsums_1d() {
        let job_coding_1: Vec<Vec<CodingType>> = vec![vec![8.0],vec![8.0],vec![32.0,175.0,128.0,128.0]];
        let job_coding_2: Vec<Vec<CodingType>> = vec![vec![8.0],vec![8.0],vec![32.0,175.0,128.0,128.0]];
        assert_eq!(get_job_qsums_1d(&job_coding_1, &job_coding_2), [(1.0,1), (1.0,1), (4.0, 4)]);
        let job_coding_1: Vec<Vec<CodingType>> = vec![vec![8.0]];
        let job_coding_2: Vec<Vec<CodingType>> = vec![vec![8.0],vec![4.0],vec![32.0,175.0,128.0,128.0]];
        assert_eq!(get_job_qsums_1d(&job_coding_1, &job_coding_2), [(1.0,1), (0.0,1), (0.0, 4)]);



    }

    #[test]
    fn test_get_job_similarity_2d() {
        let job_coding_1: Vec<Vec<Vec<CodingType>>> = vec![
            vec![],
            vec![vec![1.0]],
            vec![],
            vec![vec![4.0], vec![4.0], vec![1.0]],
            vec![],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![],
            vec![],
        ];

        let job_coding_2: Vec<Vec<Vec<CodingType>>> = vec![
            vec![],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![],
            vec![vec![1.0]],
            vec![],
            vec![],
            vec![],
            vec![],
        ];

        //println!("job similarity {:?}", job_similarity_2d(&job_coding_1, &job_coding_2));
        assert_approx_eq!(job_similarity_2d(&job_coding_1, &job_coding_2), 0.125, 0.001);

    /*
            vec![vec![2, 2, 9, 3], vec![1]],
            vec![vec![2]],

            (1, 4), (0, 1)
            
            vec![vec![2, 2, 9, 3], vec![9, 1, 1]],
            vec![vec![2, 2, 2, 2, 8, 2], vec![1], vec![8, 1, 1]],

            (3.55, 6), (0, 1), (8/9 + 1 + 1, 3)

            (3.55555555 + (8/9 + 1 + 1) + 1) / 15 = 0.4962962959259259
       */

        let job_coding_1: Vec<Vec<Vec<CodingType>>> = vec![
            vec![vec![2.0, 2.0, 9.0, 3.0], vec![1.0]],
            vec![vec![2.0, 2.0, 9.0, 3.0], vec![9.0, 1.0, 1.0]],
        ];
        let job_coding_2: Vec<Vec<Vec<CodingType>>> = vec![
            vec![vec![2.0]],
            vec![vec![2.0, 2.0, 2.0, 2.0, 8.0, 2.0], vec![1.0], vec![8.0, 1.0, 1.0]],
        ];
        //println!("job similarity {:?}", job_similarity_2d(&job_coding_1, &job_coding_2));
        assert_approx_eq!(job_similarity_2d(&job_coding_1, &job_coding_2), 0.4962, 0.001);


        /*
        job1_phase1_coding : [[2,2,2,9,3], [9,1,1,1]]
        job1_phase2_coding : [[9,3], [1]]

        job2_phase1_coding : [[1]] 
        job2_phase2_coding : [[2,2,2,2,8,2,2], [1], [8,1,1]] 

        Step 1:
        job1_phase1_coding : [[2,2,2,9,3], [9,1,1,1]]
        job2_phase1_coding : [[-,-,-,-,-], [-,1,-,-]] 
        [(0;5), (1;4)]
        sim = 1/9

        job1_phase2_coding : [[-,-,-,-,9,3,-], [1], [-,-,-]]
        job2_phase2_coding : [[2,2,2,2,8,2,2], [1], [8,1,1]] 
        [(1.55;7), (1;1), (0;3)]
        sim = (1.55 + 1 + 0)/11

        (1 + 1.55 + 1) / 21)
        */
        
        let job_coding_1 = vec![
            vec![vec![2.0,2.0,2.0,9.0,3.0], vec![9.0,1.0,1.0,1.0]],
            vec![vec![9.0,3.0], vec![1.0]],
            ];

        let job_coding_2  = vec![
            vec![vec![1.0]],
            vec![vec![2.0,2.0,2.0,2.0,8.0,2.0,2.0], vec![1.0], vec![8.0,1.0,1.0]]
        ];

        assert_approx_eq!(job_similarity_2d(&job_coding_1, &job_coding_2), 0.1775, 0.001);
    }

    #[test]
    fn test_detect_phases_1d() {
        let coding: Vec<CodingType> = vec![2.0, 2.0, 2.0, 2.0, 8.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 1.0, 1.0, 0.0];
        let res_phases = detect_phases_1d(&coding);
        let expected_phases: Vec<Vec<CodingType>> = vec![vec![2.0, 2.0, 2.0, 2.0, 8.0, 2.0], vec![1.0], vec![8.0, 1.0, 1.0]];
        assert_eq!(res_phases, expected_phases);

        let coding: Vec<CodingType> = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let res_phases = detect_phases_1d(&coding);
        let expected_phases: Vec<Vec<CodingType>> = vec![];
        assert_eq!(res_phases, expected_phases);
    }


    #[test]
    fn test_detect_phases_2d() {
        let coding = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        ];

        let expected_phases: Vec<Vec<Vec<CodingType>>> = vec![
            vec![],
            vec![vec![1.0]],
            vec![],
            vec![vec![4.0], vec![4.0], vec![1.0]],
            vec![],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![],
            vec![],
        ];

        let res_phases = detect_phases_2d(&coding);

        assert_eq!(res_phases, expected_phases);
    }
}

