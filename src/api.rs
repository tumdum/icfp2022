use crate::Move;
use reqwest::blocking::multipart;
use reqwest::blocking::Client;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write;

const KEY: &str = include_str!("../api-key");

pub fn get_all_scores() -> BTreeMap<i64, f64> {
    let client = Client::new();
    let body: Value = client
        .get("https://robovinci.xyz/api/results/user")
        .bearer_auth(KEY.trim())
        .send()
        .unwrap()
        .json()
        .unwrap();
    let mut h = BTreeMap::default();
    for obj in body["results"].as_array().unwrap() {
        let id = obj["problem_id"].as_i64().unwrap();
        let min_cost = obj["min_cost"].as_f64().unwrap();
        let min_cost = if min_cost == 0.0 {
            u32::max_value() as f64
        } else {
            min_cost
        };
        h.insert(id, min_cost);
    }
    h
}

#[derive(Debug)]
pub enum Status {
    Queued,
    Processing,
    Succeeded(f64),
    Failed,
}

pub fn submission_status(id: i64) -> Status {
    let client = Client::new();
    let body: Value = client
        .get(format!("https://robovinci.xyz/api/submissions/{id}"))
        .bearer_auth(KEY.trim())
        .send()
        .unwrap()
        .json()
        .unwrap();
    match body["status"].as_str().unwrap() {
        "QUEUED" => Status::Queued,
        "PROCESSING" => Status::Processing,
        "SUCCEEDED" => Status::Succeeded(body["cost"].as_f64().unwrap()),
        "FAILED" => Status::Failed,
        other => panic!("unknown status: {}", other),
    }
}

pub fn post_solution(id: i64, moves: &[Move]) {
    use reqwest::blocking::multipart;

    let mut output = String::new();
    for m in moves {
        writeln!(&mut output, "{}", m.to_isl());
    }
    println!("{}", output);

    let path = format!("outputs/best.{}.txt", id);
    std::fs::write(&path, output).unwrap();

    let form = multipart::Form::new().file("file", path).unwrap();

    let client = reqwest::blocking::Client::new();
    let resp : Value = client
        .post(format!("https://robovinci.xyz/api/problems/{id}"))
        .bearer_auth(KEY.trim())
        .multipart(form)
        .send()
        .unwrap()
        .json()
        .unwrap();
    let sub_id = resp["submission_id"].as_i64().unwrap();
    loop {
        let status = dbg!(submission_status(sub_id));
        match status {
            Status::Succeeded(cost) => {
                println!("Computed cost: {}", cost);
                break;
            }
            Status::Failed => panic!(),
            _ => {},
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
