use crate::*;
use anyhow::{bail, ensure, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

const DIRS: [Dir; 2] = [Dir::X, Dir::Y];
const DEPTH: i64 = 50;
const FAN_OUT_START: i64 = 40;
const KEEP_SIZE: i64 = 600;
const REDUCTION: i64 = 1;

#[derive(Clone)]
pub struct Solver {
    state: State,
    known: HashMap<Vec<Move>, HashMap<Name, (Shape, Color)>>,
    rng: SmallRng,
}

impl Solver {
    pub fn new(state: State) -> Self {
        Self {
            state,
            known: Default::default(),
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn random_name<'a>(&mut self, names: &'a [Name]) -> &'a Name {
        &names[self.rng.gen_range(0..names.len())]
    }

    fn random_dir(&mut self) -> Dir {
        DIRS[self.rng.gen_range(0..DIRS.len())]
    }

    fn range_for_dir(&self, dir: Dir, target: &Name, shape: &Shape) -> Range<i32> {
        match dir {
            Dir::X => (shape.l..(shape.l + shape.w)),
            Dir::Y => (shape.b..(shape.b + shape.h)),
        }
    }

    pub fn random_move(
        &mut self,
        blocks: &HashMap<Name, (Shape, Color)>,
        last: &Move,
    ) -> Result<Move> {
        // TODO point
        for _ in 0..5 {
            let cand = match self.rng.gen_range(0..3) {
                0 => self.random_line_cut(blocks)?,
                1 => self.random_color(blocks)?,
                _ => self.random_point_cut(blocks)?,
            };

            if &cand != last {
                return Ok(cand);
            }
        }
        bail!("can't find")
    }

    fn random_color(&mut self, blocks: &HashMap<Name, (Shape, Color)>) -> Result<Move> {
        let names: Vec<_> = blocks.keys().cloned().collect();
        let target_name = self.random_name(&names);
        let target_color = self.best_color_for(&target_name, blocks)?;
        Ok(Move {
            block: target_name.clone(),
            kind: Kind::Color(target_color),
        })
    }

    fn best_color_for(&self, name: &Name, block: &HashMap<Name, (Shape, Color)>) -> Result<Color> {
        let shape = block.get(name).unwrap().0;
        shape.avg_color_in(&self.state.target)
    }

    fn random_point_cut(&mut self, blocks: &HashMap<Name, (Shape, Color)>) -> Result<Move> {
        let names: Vec<_> = blocks.keys().cloned().collect();
        let target = self.random_name(&names);
        let shape = blocks.get(target).unwrap().0;

        let dir_x_range = self.range_for_dir(Dir::X, target, &shape);
        let dir_y_range = self.range_for_dir(Dir::Y, target, &shape);
        ensure!(!dir_x_range.is_empty());
        ensure!(!dir_y_range.is_empty());
        let cut_point_x = self.rng.gen_range(dir_x_range);
        let cut_point_y = self.rng.gen_range(dir_y_range);

        assert!(cut_point_x >= shape.l);
        assert!(cut_point_x < (shape.l + shape.w));
        assert!(cut_point_y >= shape.b);
        assert!(cut_point_y < (shape.b + shape.h));

        Ok(Move {
            block: target.clone(),
            kind: Kind::PointCut {
                x: cut_point_x,
                y: cut_point_y,
            },
        })
    }

    fn random_line_cut(&mut self, blocks: &HashMap<Name, (Shape, Color)>) -> Result<Move> {
        let names: Vec<_> = blocks.keys().cloned().collect();
        let target = self.random_name(&names);
        let dir: Dir = self.random_dir();
        let shape = blocks.get(target).unwrap().0;
        let dir_range = self.range_for_dir(dir, target, &shape);
        ensure!(!dir_range.is_empty());
        let cut_point = self.rng.gen_range(dir_range);
        if dir == Dir::X {
            assert!(cut_point >= shape.l);
            assert!(cut_point < (shape.l + shape.w));
        } else {
            assert!(cut_point >= shape.b);
            assert!(cut_point < (shape.b + shape.h));
        }
        Ok(Move {
            block: target.clone(),
            kind: Kind::LineCut(dir, cut_point),
        })
    }

    fn blocks_of(&mut self, base: &[Move]) -> HashMap<Name, (Shape, Color)> {
        match self.known.entry(base.to_vec()) {
            Occupied(e) => e.get().clone(),
            Vacant(e) => {
                let (block, _, _) = self.state.run(e.key()).unwrap();
                let h = block.to_hash();
                e.insert(h.clone());
                h
            }
        }
    }

    fn score_of(&mut self, moves: &[Move]) -> f64 {
        self.state.run(moves).unwrap().2
    }

    pub fn new_solutions_from(&mut self, base: &[Move], i: i64) -> Vec<(Vec<Move>, f64)> {
        let max = FAN_OUT_START; // - i;
        let max = if max < 10 {
            10
        } else {
            max
        };
        let ret = (0..max)
            .flat_map(|_| {
                let blocks = self.blocks_of(base);
                let mut base = base.to_vec();
                let last = base.last().unwrap();
                base.push(self.random_move(&blocks, last).ok()?);
                let score = self.score_of(&base);
                Some((base, score))
            })
            .collect();

        ret
    }

    pub fn new_solutions_from_n(&mut self, base: &[Move], n: i64) -> Vec<(Vec<Move>, f64)> {
        let tmp_score = self.score_of(base);
        let mut solutions = vec![(base.to_vec(), tmp_score)];
        for i in 0..n {
            let mut new_solutions: Vec<(Vec<Move>, f64)> = solutions
                .par_iter()
                .flat_map(|(base, _)| self.clone().new_solutions_from(base, i as i64))
                .collect();
            eprintln!("new solutions: {}", new_solutions.len());
            new_solutions.append(&mut solutions);
            new_solutions.sort_by_key(|v| v.1 as usize);
            use rand::prelude::SliceRandom;

            if new_solutions.len() > KEEP_SIZE as usize{
                let mut best = new_solutions[..((KEEP_SIZE/2) as usize)].to_vec();
                let mut random = new_solutions[((KEEP_SIZE/2) as usize)..]
                    .choose_multiple(&mut self.rng, (KEEP_SIZE/2) as usize)
                    .cloned()
                    .collect();

                solutions = vec![];
                solutions.append(&mut best);
                solutions.append(&mut random);
            } else {
                new_solutions.truncate(KEEP_SIZE as usize);
                solutions = new_solutions;
            }
            assert!(solutions.len() > 0);
            eprintln!(
                "{}: solutions#: {} (best: {})",
                i,
                solutions.len(),
                solutions[0].1
            );
        }
        solutions
    }

    pub fn find_solutions(&mut self) -> Result<(Vec<Move>, f64)> {
        let full_box = Shape {
            l: 0,
            b: 0,
            h: self.state.target.len() as i32,
            w: self.state.target[0].len() as i32,
        };
        let best_color = full_box.avg_color_in(&self.state.target)?;
        let moves = vec![Move {
            block: "0".into(),
            kind: Kind::Color(best_color),
        }];

        let mut candidates = self.new_solutions_from_n(&moves, DEPTH);
        candidates.sort_by_key(|v| v.1 as usize);
        Ok(candidates[0].clone())
    }
}
