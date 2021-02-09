pub type PhaseNum = u16;
pub type PhaseLength = PhaseNum;




#[derive(Debug, PartialEq)]
pub struct Combination {
    combi: Vec<u8>,
}

impl Combination {
    pub fn new(combi: Vec<u8>) -> Combination {
        Combination {
            combi,
        }
    }

    //pub fn get_combination(&self) -> Vec<u8> {
    //    self.combi.clone()
    //}

    pub fn to_indices_attached(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = Vec::new();
        for i in 0..self.combi.len() {
            if self.combi[i] == 1 {
                indices.push(i as usize);
            }
        }
        indices
    }

    pub fn to_indices_not_attached(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = Vec::new();
        for i in 0..self.combi.len() {
            if self.combi[i] == 0 {
                indices.push(i as usize);
            }
        }
        indices
    }
}


fn shift(combi: &Vec<u8>) -> Option<Vec<u8>> {
    let mut res = combi.clone();
    let size = combi.len();
    let mut pos = 0;

    // Termination: If all one are on the left side and zeros on the right side, then terminate
    while (pos < size) && (combi[pos] == 1) {
        pos += 1;
    }
    while (pos < size) && (combi[pos] == 0) {
        pos += 1;
    }
    if pos == size {
        return None;
    }
    
    // Rule 1: if first position is zero, then move first occurence of 1 to the left
    if res[0] == 0 {
        res[pos] = 0;
        res[pos-1] = 1;
        return Some(res)
    }

    // Rule 2: 
    //  1. Skip leading ones
    //  2. Skip zeros after leading ones
    //  3. Move first occurence of one to the left
    //  4. Move leading ones to right, to the first occures of one
    if res[0] == 1 {
        let mut c1 = 0;
        while (c1 < size) && (res[c1] == 1) {
            res[c1] = 0;
            c1 += 1;
        }
        let mut c2 = c1;
        while (c2 < size) && (res[c2] == 0) {
            c2 += 1;
        }
        res[c2] = 0;
        res[c2-1] = 1;
        for i in (0..(c1+1)).rev() {
            res[c2-1-i] = 1;
        }
    }
    Some(res)
}


pub struct PhaseCombinations {
    v: Option<Vec<u8>>,
}

impl PhaseCombinations {
    pub fn new(l1: PhaseNum, l2: PhaseNum) -> PhaseCombinations{
        assert!(l1 <= l2);
        let mut combi = vec![1; std::cmp::max(l1 as usize, l2 as usize)];
        //for i in 0..std::cmp::min(l1, l2) {
        for i in 0..(l2-l1) {
            combi[i as usize] = 0;
        }

        PhaseCombinations {
            v: Some(combi),
        }
    }
}

impl Iterator for PhaseCombinations {
    type Item = Combination;

    fn next(&mut self) -> Option<Self::Item> {
        let tmp = self.v.clone();
        match tmp {
            Some(combi) => {
                self.v = shift(&combi);
                Some(Combination::new(combi))
            }

            None => None
        }
    }

}


#[cfg(test)]
mod tests {
    //use assert_approx_eq::assert_approx_eq;
    use super::*;

    #[test]
    fn test_termination() {
        let combi = vec![1, 1, 1, 0, 0, 0];
        let s = shift(&combi);
        assert_eq!(s, None);
    }

    #[test]
    fn test_shift() {
        // expected, returned
        assert_eq!(Some(vec![0, 0, 1, 0, 1, 1]), shift(&vec![0, 0, 0, 1, 1, 1]));
        assert_eq!(Some(vec![0, 1, 0, 0, 1, 1]), shift(&vec![0, 0, 1, 0, 1, 1]));
        assert_eq!(Some(vec![1, 0, 0, 0, 1, 1]), shift(&vec![0, 1, 0, 0, 1, 1]));
        assert_eq!(Some(vec![0, 0, 1, 1, 0, 1]), shift(&vec![1, 0, 0, 0, 1, 1]));
        assert_eq!(Some(vec![0, 1, 0, 1, 0, 1]), shift(&vec![0, 0, 1, 1, 0, 1]));
        assert_eq!(Some(vec![1, 0, 0, 1, 0, 1]), shift(&vec![0, 1, 0, 1, 0, 1]));
        assert_eq!(Some(vec![0, 1, 1, 0, 0, 1]), shift(&vec![1, 0, 0, 1, 0, 1]));
        assert_eq!(Some(vec![1, 0, 1, 0, 0, 1]), shift(&vec![0, 1, 1, 0, 0, 1]));
        assert_eq!(Some(vec![1, 1, 0, 0, 0, 1]), shift(&vec![1, 0, 1, 0, 0, 1]));
        assert_eq!(Some(vec![0, 0, 1, 1, 1, 0]), shift(&vec![1, 1, 0, 0, 0, 1]));
        assert_eq!(Some(vec![0, 1, 0, 1, 1, 0]), shift(&vec![0, 0, 1, 1, 1, 0]));
        assert_eq!(Some(vec![1, 0, 0, 1, 1, 0]), shift(&vec![0, 1, 0, 1, 1, 0]));
        assert_eq!(Some(vec![0, 1, 1, 0, 1, 0]), shift(&vec![1, 0, 0, 1, 1, 0]));
        assert_eq!(Some(vec![1, 0, 1, 0, 1, 0]), shift(&vec![0, 1, 1, 0, 1, 0]));
        assert_eq!(Some(vec![1, 1, 0, 0, 1, 0]), shift(&vec![1, 0, 1, 0, 1, 0]));
        assert_eq!(Some(vec![1, 0, 1, 1, 0, 0]), shift(&vec![0, 1, 1, 1, 0, 0]));
        assert_eq!(Some(vec![1, 1, 0, 1, 0, 0]), shift(&vec![1, 0, 1, 1, 0, 0]));
        assert_eq!(Some(vec![1, 1, 1, 0, 0, 0]), shift(&vec![1, 1, 0, 1, 0, 0]));
        assert_eq!(None, shift(&vec![1, 1, 1, 0, 0, 0]));
    }

    #[test]
    fn test_struct() {
        let mut pcs_iter = PhaseCombinations::new(1, 2).into_iter();
        println!("{:?}", pcs_iter.next().unwrap());
        println!("{:?}", pcs_iter.next().unwrap());
        assert_eq!(None, pcs_iter.next());
    }

    #[test]
    fn test_zero_phases() {
        let mut pcs_iter = PhaseCombinations::new(0, 2).into_iter();
        println!("{:?}", pcs_iter.next().unwrap());
        assert_eq!(None, pcs_iter.next());
    }

    //#[test]
    //fn test_iterator() {
    //    let mut pcs_iter = PhaseCombinations::new(3, 6).into_iter();
    //        assert_eq!(vec![0, 0, 0, 1, 1, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 0, 1, 0, 1, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 1, 0, 0, 1, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 0, 0, 0, 1, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 0, 1, 1, 0, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 1, 0, 1, 0, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 0, 0, 1, 0, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 1, 1, 0, 0, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 0, 1, 0, 0, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 1, 0, 0, 0, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 0, 1, 1, 1, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 1, 0, 1, 1, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 0, 0, 1, 1, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 1, 1, 0, 1, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 0, 1, 0, 1, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 1, 0, 0, 1, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![0, 1, 1, 1, 0, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 0, 1, 1, 0, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 1, 0, 1, 0, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(vec![1, 1, 1, 0, 0, 0], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(None, pcs_iter.next());
    //}


    //#[test]
    //fn test_iterator_edge_cases() {
    //    let mut pcs_iter = PhaseCombinations::new(0, 0).into_iter();
    //        let empty: Vec<u8> = vec![];
    //        assert_eq!(empty, pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(None, pcs_iter.next());
    //    let mut pcs_iter = PhaseCombinations::new(1, 1).into_iter();
    //        assert_eq!(vec![1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(None, pcs_iter.next());
    //    let mut pcs_iter = PhaseCombinations::new(2, 2).into_iter();
    //        assert_eq!(vec![1, 1], pcs_iter.next().unwrap().get_combination());
    //        assert_eq!(None, pcs_iter.next());
    //}
}
