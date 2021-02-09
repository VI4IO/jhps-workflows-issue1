use std::collections::HashMap;

pub type Jobid = u32;
//type HexCoding = Vec<Vec<algorithm::CodingType>>;
//pub type HexCoding = Vec<Vec<hex::Score>>;
//pub type AbsCoding = Vec<algorithm2::CodingType>;
//pub type AbsAggzerosCoding = Vec<algorithm2::CodingType>;
//pub type PhasesCoding= Vec<Vec<Vec<algorithm::CodingType>>>;
//pub type PhasesFloatCoding= Vec<Vec<Vec<algorithm::CodingType>>>;

#[derive(Clone)]
pub struct Profile<T> {
    pub name: String,
    pub id: u32,
    pub dataset: HashMap<Jobid,T>,
    pub func: fn(&T, &T) -> f32,
}

impl<T> std::fmt::Debug for Profile<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("").finish()
    }
}

//#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize)]
#[derive(Debug, Clone)]
pub enum ALG {
    //Float(Profile<float::JobCoding>),
    Bin(Profile<bin::JobCoding>),
    Hex(Profile<hex::JobCoding>),
}

#[derive(Debug, Clone, Copy)]
pub struct Entity {
    pub jobid: u32,
    pub sim: f32,
}

pub struct Cluster {
    pub centroid_jobid: u32,
    pub entities: Vec<Entity>,
}

pub struct ClusterCentroid<T> {
    pub centroid_jobid: u32,
    pub centroid_coding: T,
    pub entities: Vec<Entity>,
}
