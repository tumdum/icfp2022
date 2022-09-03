use serde::{Deserialize, Serialize};
use crate::solver::Solver;
use anyhow::{ensure, Result};
use image::io::Reader as ImageReader;
use image::{GenericImageView, Rgba};
use smol_str::SmolStr;
use std::collections::HashMap;
use std::path::PathBuf;
use structopt::StructOpt;
use std::fs::File;
use serde_json::Value;

mod api;
mod solver;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

#[derive(Hash, Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    b: i32,
    l: i32,
    w: i32,
    h: i32,
}

impl Shape {
    const fn new(l: i32, b: i32, w: i32, h: i32) -> Self {
        Self { l, b, w, h }
    }

    const fn size(&self) -> usize {
        self.w as usize * self.h as usize
    }

    fn avg_color_in(&self, img: &[Vec<Color>]) -> Result<Color> {
        let mut r = vec![];
        let mut g = vec![];
        let mut b = vec![];
        let mut a = vec![];

        for col in self.b..(self.b + self.h) {
            for row in self.l..(self.l + self.w) {
                r.push(img[col as usize][row as usize].r);
                g.push(img[col as usize][row as usize].g);
                b.push(img[col as usize][row as usize].b);
                a.push(img[col as usize][row as usize].a);
            }
        }

        r.sort();
        g.sort();
        b.sort();
        a.sort();
        let l = r.len();
        ensure!(l > 0);
        Ok(Color {
            r: (r.into_iter().map(|v| v as usize).sum::<usize>() / l) as u8,
            g: (g.into_iter().map(|v| v as usize).sum::<usize>() / l) as u8,
            b: (b.into_iter().map(|v| v as usize).sum::<usize>() / l) as u8,
            a: (a.into_iter().map(|v| v as usize).sum::<usize>() / l) as u8,
        })
    }
}

#[derive(Hash, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl From<Rgba<u8>> for Color {
    fn from(c: Rgba<u8>) -> Self {
        Self {
            r: c[0],
            g: c[1],
            b: c[2],
            a: c[3],
        }
    }
}

impl Color {
    const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

#[derive(Deserialize, Hash, Debug, Clone, PartialEq, Eq)]
pub struct Name(SmolStr);

impl Name {
    fn add(&self, val: i32) -> Self {
        Self(SmolStr::new(&format!("{}.{}", self.0, val)))
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Name(SmolStr::new_inline(s))
    }
}

#[derive(Hash, Debug, Clone, PartialEq)]
pub enum Block {
    Simple(Name, Shape, Color),
    Complex(Vec<Block>),
}

impl Block {
    fn default(w: i32, h: i32) -> Self {
        let shape = Shape {
            l: 0,
            b: 0,
            w,
            h,
        };
        Block::Simple(
            "0".into(),
            shape,
            Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
        )
    }
    fn to_vec(self) -> Vec<Block> {
        match self {
            Self::Complex(blocks) => blocks,
            _ => unreachable!(),
        }
    }

    fn set_name(&mut self, name: Name) {
        match self {
            Self::Simple(_, shape, color) => {
                *self = Self::Simple(name, *shape, *color);
            }
            _ => unreachable!(),
        }
    }

    fn to_hash(&self) -> HashMap<Name, (Shape, Color)> {
        match self {
            Self::Simple(name, shape, color) => {
                let mut h = HashMap::default();
                h.insert(name.clone(), (shape.clone(), color.clone()));
                h
            }
            Self::Complex(blocks) => {
                let mut h = HashMap::default();
                for b in blocks {
                    for (k, v) in b.to_hash() {
                        h.insert(k, v);
                    }
                }
                h
            }
        }
    }

    fn flatten(&self) -> Vec<(Shape, Color)> {
        match self {
            Self::Simple(_, shape, color) => vec![(*shape, *color)],
            Self::Complex(blocks) => blocks.iter().flat_map(|b| b.flatten()).collect(),
        }
    }

    fn to_image(&self, w: usize, h: usize) -> Vec<Vec<Color>> {
        let row = vec![Color::default(); w];
        let mut output: Vec<Vec<Color>> = vec![row; h];
        let shapes = self.flatten();
        for (shape, color) in shapes {
            for x in shape.l..(shape.l + shape.w) {
                for y in shape.b..(shape.b + shape.h) {
                    output[y as usize][x as usize] = color;
                }
            }
        }
        output
    }
}

#[derive(Hash, Debug, Clone, PartialEq, Eq)]
pub struct Move {
    block: Name,
    kind: Kind,
}

impl Move {
    fn base_cost(&self) -> f64 {
        match self.kind {
            Kind::LineCut { .. } => 7.0,
            Kind::PointCut { .. } => 10.0,
            Kind::Color { .. } => 5.0,
            Kind::Swap { .. } => 3.0,
            Kind::Merge { .. } => 1.0,
        }
    }

    fn to_isl(&self) -> String {
        match self.kind {
            Kind::LineCut(dir, amount) => {
                format!("cut [{}] [{}] [{}]", self.block.0, dir, amount)
            }
            Kind::PointCut { x, y } => format!("cut [{}] [{},{}]", self.block.0, x, y),
            Kind::Color(Color { r, g, b, a }) => {
                format!("color [{}] [{},{},{},{}]", self.block.0, r, g, b, a)
            }
            Kind::Swap { .. } => todo!(),
            Kind::Merge { .. } => todo!(),
        }
    }
}

#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    X,
    Y,
}

impl std::fmt::Display for Dir {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        match self {
            Dir::X => f.write_str("X"),
            Dir::Y => f.write_str("Y"),
        }
    }
}

#[derive(Hash, Debug, Clone, PartialEq, Eq)]
enum Kind {
    LineCut(Dir, i32),
    PointCut { x: i32, y: i32 },
    Color(Color),
    Swap(Name),
    Merge(Name),
}

#[derive(Clone)]
pub struct State {
    w: usize,
    h: usize,
    target: Vec<Vec<Color>>,
    init_block: Block,
}

impl State {
    fn default() -> Self {
        Self {
            w: 400,
            h: 400,
            target: vec![],
            init_block: Block::default(400, 400),
        }
    }
    fn from_img_view(v: impl GenericImageView<Pixel = Rgba<u8>>) -> Self {
        let (w, h) = v.dimensions();
        let row = vec![Color::default(); w as usize];
        let mut target: Vec<Vec<Color>> = vec![row; h as usize];
        for x in 0..w {
            for y in 0..h {
                let pixel = v.get_pixel(x, y);
                target[(h - y) as usize - 1][x as usize] = pixel.into();
            }
        }
        let shape = Shape {
            l: 0,
            b: 0,
            w: w as i32,
            h: h as i32,
        };
        Self {
            w: w as usize,
            h: h as usize,
            target,
            init_block: Block::default(w as i32,h as i32),
        }
    }

    pub fn set_init_block(&mut self, init: Block) {
        self.init_block = init;
    }

    fn run(&mut self, moves: &[Move]) -> Result<(Block, Vec<Vec<Color>>, f64)> {
        let shape = Shape {
            l: 0,
            b: 0,
            w: self.w as i32,
            h: self.h as i32,
        };
        /*
        let mut current = Block::Simple(
            "0".into(),
            shape,
            Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
        );
        */
        let mut current = self.init_block.clone();

        // dbg!(shape.avg_color_in(&self.target));

        let mut moves_costs = 0;

        for m in moves {
            let (block, cost) = self.apply_with_cost(current, m)?;
            moves_costs += cost;
            current = block;
        }

        let output = current.to_image(self.w, self.h);
        let score = image_diff(&output, &self.target) + moves_costs as f64;

        Ok((current, output, score))
    }

    fn canvas_size(&self) -> f64 {
        (self.w * self.h) as f64
    }

    fn apply(&mut self, b: Block, m: &Move) -> Result<Block> {
        Ok(self.apply_inner(b, m)?.0)
    }

    fn apply_with_cost(&mut self, b: Block, m: &Move) -> Result<(Block, usize)> {
        let (block, block_size) = self.apply_inner(b, m)?;
        let cost = m.base_cost() * (self.canvas_size() / block_size.unwrap() as f64);
        Ok((block, cost.round() as usize))
    }

    fn apply_inner(&mut self, b: Block, m: &Move) -> Result<(Block, Option<usize>)> {
        match b {
            Block::Simple(ref name, shape, color) if name == &m.block => match m.kind {
                Kind::LineCut(Dir::X, amount) => {
                    // ensure!(amount < shape.w);
                    ensure!(amount >= shape.l);
                    ensure!(amount < (shape.l + shape.w));
                    let left_w = amount - shape.l;
                    let left = Block::Simple(name.add(0), Shape { w: left_w, ..shape }, color);
                    let right = Block::Simple(
                        name.add(1),
                        Shape {
                            w: shape.l + shape.w - amount,
                            l: amount,
                            ..shape
                        },
                        color,
                    );
                    Ok((Block::Complex(vec![left, right]), Some(shape.size())))
                }
                Kind::LineCut(Dir::Y, amount) => {
                    // ensure!(amount < shape.h);
                    ensure!(amount >= shape.b);
                    ensure!(amount < (shape.b + shape.h));
                    let top = Block::Simple(
                        name.add(1),
                        Shape {
                            b: amount,
                            h: shape.b + shape.h - amount,
                            ..shape
                        },
                        color,
                    );
                    let bottom = Block::Simple(
                        name.add(0),
                        Shape {
                            h: amount - shape.b,
                            ..shape
                        },
                        color,
                    );
                    Ok((Block::Complex(vec![top, bottom]), Some(shape.size())))
                }
                Kind::PointCut { x, y } => {
                    let tmp = self.apply(
                        b.clone(),
                        &Move {
                            block: m.block.clone(),
                            kind: Kind::LineCut(Dir::Y, y),
                        },
                    );

                    let mut blocks = tmp?.to_vec();
                    let bottom = blocks.remove(1);
                    let top = blocks.remove(0);

                    let tmp = self.apply(
                        top,
                        &Move {
                            block: m.block.add(1),
                            kind: Kind::LineCut(Dir::X, x),
                        },
                    );
                    let mut blocks = tmp?.to_vec();
                    let mut top_right = blocks.remove(1);
                    let mut top_left = blocks.remove(0);

                    let tmp = self.apply(
                        bottom,
                        &Move {
                            block: m.block.add(0),
                            kind: Kind::LineCut(Dir::X, x),
                        },
                    );
                    let mut blocks = tmp?.to_vec();
                    let mut bottom_right = blocks.remove(1);
                    let mut bottom_left = blocks.remove(0);

                    top_left.set_name(name.add(3));
                    top_right.set_name(name.add(2));
                    bottom_left.set_name(name.add(0));
                    bottom_right.set_name(name.add(1));
                    Ok((
                        Block::Complex(vec![top_left, top_right, bottom_right, bottom_left]),
                        Some(shape.size()),
                    ))
                }
                Kind::Color(c) => Ok((Block::Simple(name.clone(), shape, c), Some(shape.size()))),
                _ => todo!(),
            },
            Block::Simple { .. } => Ok((b, None)),
            Block::Complex(blocks) => {
                let blocks: Result<Vec<(Block, Option<usize>)>> =
                    blocks.into_iter().map(|b| self.apply_inner(b, m)).collect();
                let (blocks, sizes): (Vec<Block>, Vec<Option<usize>>) = blocks?.into_iter().unzip();
                debug_assert!(sizes.iter().filter(|s| s.is_some()).count() <= 1);

                Ok((Block::Complex(blocks), sizes.into_iter().find_map(|v| v)))
            }
        }
    }
}

fn image_diff(f1: &[Vec<Color>], f2: &[Vec<Color>]) -> f64 {
    let mut diff = 0f64;
    let alpha = 0.005;

    assert_eq!(f1.len(), f2.len());

    for y in 0..f1.len() {
        assert_eq!(f1[y].len(), f2[y].len());
        for x in 0..f1[y].len() {
            let p1 = f1[y][x];
            let p2 = f2[y][x];
            diff += pixel_diff(p1, p2);
        }
    }

    (diff * alpha).round()
}

fn pixel_diff(p1: Color, p2: Color) -> f64 {
    let rdist = p1.r as f64 - p2.r as f64;
    let rdist = rdist * rdist;

    let gdist = p1.g as f64 - p2.g as f64;
    let gdist = gdist * gdist;

    let bdist = p1.b as f64 - p2.b as f64;
    let bdist = bdist * bdist;

    let adist = p1.a as f64 - p2.a as f64;
    let adist = adist * adist;

    (rdist + gdist + bdist + adist).sqrt()
}

#[derive(Debug, Deserialize)]
struct JsonBlock {
    blockId: Name,

    bottomLeft: Vec<i32>,
    topRight: Vec<i32>,
    color: Vec<u8>,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    eprintln!("{:?}", opt);
    let best = crate::api::get_all_scores();
    let id: i64 = opt.input.file_stem().unwrap().to_str().unwrap().parse()?;
    let score_to_beat = best.get(&id).unwrap();
    eprintln!("Best score for {}: {}", id, score_to_beat);
    let img = ImageReader::open(opt.input.clone())?.decode()?;

    let init_json_path = opt.input.parent().unwrap();
    let mut init_json_path = init_json_path.to_path_buf();
    init_json_path.push(format!("{id}.initial.json"));
    let init_json : Value = serde_json::from_reader(&File::open(init_json_path.clone())?)?;
    let mut init_blocks = vec![];
    for block in init_json["blocks"].as_array().unwrap() {
        let jb : JsonBlock = serde_json::from_value(block.clone())?;
        let c = Color{
            r: jb.color[0],
            g: jb.color[1],
            b: jb.color[2],
            a: jb.color[3],
        };
        let s = Shape{
            l: jb.bottomLeft[0],
            b: jb.bottomLeft[1],
            w: jb.topRight[0] - jb.bottomLeft[0],
            h: jb.topRight[1] - jb.bottomLeft[1],
        };
        init_blocks.push(Block::Simple(jb.blockId,
            s, c));
    }

    let mut state = State::from_img_view(img);
    if !init_blocks.is_empty() {
        eprintln!("Using starting block from {:?}", init_json_path);
        state.set_init_block(Block::Complex(init_blocks));
    }

    let mut solver = Solver::new(state);
    let moves = solver.find_solutions()?;

    if moves.1 < *score_to_beat {
        let diff = *score_to_beat - moves.1;
        println!(
            "Found improvment by {} ({}%): {} -> {}",
            diff,
            100.0 * (diff / *score_to_beat),
            score_to_beat,
            moves.1
        );
        crate::api::post_solution(id, &moves.0);
    } else {
        println!(
            "Solution found ({}) is worse than submitted one ({}) by {}",
            moves.1,
            score_to_beat,
            moves.1 - *score_to_beat
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    const BG: Color = Color::new(255, 255, 255, 255);

    #[test]
    fn line_cut_test() -> Result<()> {
        let b = Block::Simple("0".into(), Shape::new(0, 0, 100, 100), BG);

        let m = Move {
            block: "0".into(),
            kind: Kind::LineCut(Dir::X, 40),
        };

        let expected = Block::Complex(vec![
            Block::Simple(
                "0.0".into(),
                Shape {
                    b: 0,
                    l: 0,
                    w: 40,
                    h: 100,
                },
                BG,
            ),
            Block::Simple(
                "0.1".into(),
                Shape {
                    b: 0,
                    l: 40,
                    w: 60,
                    h: 100,
                },
                BG,
            ),
        ]);

        assert_eq!(expected, State::default().apply(b, &m)?);

        let b = Block::Simple("4".into(), Shape::new(0, 0, 100, 100), BG);

        let m = Move {
            block: "4".into(),
            kind: Kind::LineCut(Dir::Y, 60),
        };

        let expected = Block::Complex(vec![
            Block::Simple(
                "4.1".into(),
                Shape {
                    l: 0,
                    b: 60,
                    w: 100,
                    h: 40,
                },
                BG,
            ),
            Block::Simple(
                "4.0".into(),
                Shape {
                    l: 0,
                    b: 0,
                    w: 100,
                    h: 60,
                },
                BG,
            ),
        ]);

        assert_eq!(expected, State::default().apply(b, &m)?);

        Ok(())
    }

    #[test]
    fn point_cut_test() -> Result<()> {
        let b = Block::Complex(vec![
            Block::Simple(
                "1".into(),
                Shape {
                    l: 0,
                    b: 0,
                    w: 40,
                    h: 50,
                },
                BG,
            ),
            Block::Simple(
                "2".into(),
                Shape {
                    l: 0,
                    b: 50,
                    w: 40,
                    h: 50,
                },
                BG,
            ),
            Block::Simple(
                "3".into(),
                Shape {
                    l: 40,
                    b: 0,
                    w: 60,
                    h: 100,
                },
                BG,
            ),
        ]);

        let m = Move {
            block: "1".into(),
            kind: Kind::PointCut { x: 30, y: 30 },
        };
        let expected = Block::Complex(vec![
            Block::Complex(vec![
                Block::Simple(
                    "1.3".into(),
                    Shape {
                        l: 0,
                        b: 30,
                        w: 30,
                        h: 20,
                    },
                    BG,
                ),
                Block::Simple(
                    "1.2".into(),
                    Shape {
                        l: 30,
                        b: 30,
                        w: 10,
                        h: 20,
                    },
                    BG,
                ),
                Block::Simple(
                    "1.1".into(),
                    Shape {
                        l: 30,
                        b: 0,
                        w: 10,
                        h: 30,
                    },
                    BG,
                ),
                Block::Simple(
                    "1.0".into(),
                    Shape {
                        l: 0,
                        b: 0,
                        w: 30,
                        h: 30,
                    },
                    BG,
                ),
            ]),
            Block::Simple(
                "2".into(),
                Shape {
                    l: 0,
                    b: 50,
                    w: 40,
                    h: 50,
                },
                BG,
            ),
            Block::Simple(
                "3".into(),
                Shape {
                    l: 40,
                    b: 0,
                    w: 60,
                    h: 100,
                },
                BG,
            ),
        ]);

        assert_eq!(expected, State::default().apply(b, &m)?);

        Ok(())
    }

    #[test]
    fn score() -> Result<()> {
        let img = ImageReader::open("inputs/1.png")?.decode()?;
        let mut state = State::from_img_view(img);

        let code = vec![
            /*
                color [0] [85, 35, 74, 212]
                cut [0] [y] [75]
                color [0.1] [63, 156, 147, 134]
            */
            Move {
                block: "0".into(),
                kind: Kind::Color(Color {
                    r: 85,
                    g: 35,
                    b: 74,
                    a: 212,
                }),
            },
            Move {
                block: "0".into(),
                kind: Kind::LineCut(Dir::Y, 75),
            },
            Move {
                block: "0.1".into(),
                kind: Kind::Color(Color {
                    r: 63,
                    g: 156,
                    b: 147,
                    a: 134,
                }),
            },
        ];

        assert_eq!(186311, state.run(&code)?.2 as usize);

        Ok(())
    }
}
