fn main() {
    let filename = "depression_data.csv";
    let people_vector = read_csv(filename);
    let mut individuals_vector = vec![];
    for person in people_vector {
        individuals_vector.push(to_individual(person));
    }

}

use serde::Deserialize;
use std::path::Path;
// use ndarray::Array2;
use rand::Rng;
use std::fs;
use csv;
use std::error::Error;
use std::fmt::Display;

fn read_csv<P: AsRef<Path> + std::fmt::Display>(filename: P) -> Vec<Person> {
    // Check if the file exists before trying to open it
    if fs::metadata(filename).is_err() {
        eprintln!("The file '{}' does not exist or cannot be accessed.", filename);
        return Err("File not found".into());
    }
    let mut rdr = csv::Reader::from_path(filename)?;
    println!("length: {:?}", rdr.len());
    let mut vector_of_people:Vec<Person> = Vec::new();


    for (index,row) in rdr.deserialize::<Vec<String>>().enumerate() {
        if index != 0 {
            let unwrapped = row.unwrap().split();
            let name = unwrapped[0].to_string();
            let age = unwrapped[1].parse::<usize>().unwrap();
            let marital = unwrapped[2].to_string();
            let education = unwrapped[3].to_string();
            let children = unwrapped[4].parse::<usize>().unwrap();
            let smoke = unwrapped[5].to_string();
            let activity = unwrapped[5].to_string();
            let mut employment = false;
            if unwrapped[6] == "Employed".to_string() {
                employment = true;
            }
            let income = unwrapped[7].parse::<f32>().unwrap();
            let alcohol = unwrapped[8].to_string();
            let diet = unwrapped[9].to_string();
            let sleep = unwrapped[10].to_string();
            let mut mental_illness = false;
            if unwrapped[11] == "Yes".to_string() {
                mental_illness = true;
            }
            let mut substance_abuse = false;
            if unwrapped[12] == "Yes".to_string() {
                substance_abuse = true;
            }
            let mut family_history_depression = false;
            if unwrapped[13] == "Yes".to_string() {
                family_history_depression = true;
            }
            let mut chronic_condition = false;
            if unwrapped[14] == "Yes".to_string() {
                chronic_condition = true;
            }

            let person = Person{name:name, age:age, marital:marital, education:education, children:children, smoke:smoke, activity:activity, employment:employment, income:income, alcohol:alcohol, diet:diet, sleep:sleep, mental_illness:mental_illness, substance_abuse:substance_abuse, family_history_depression:family_history_depression, chronic_condition:chronic_condition};
            vector_of_people.push(person);
        }
    }
    return vector_of_people;
}

struct Person{
    name: String,
    age: usize,
    marital: String,
    education: String,
    children: usize,
    smoke: String,
    activity: String,
    employment: bool,
    income: f32,
    alcohol: String,
    diet: String,
    sleep: String,
    mental_illness: bool,
    substance_abuse: bool,
    family_history_depression: bool,
    chronic_condition: bool,
}

fn to_individual(person:Person) -> Individual {
    let age = person.age as f32;
    let mut marital = 0.0;
    match person.marital.as_str() {
        "Single" => marital = 1.0,
        "Married" => marital = 2.0,
        "Divorced" => marital = 3.0,
        "Widowed" => marital = 4.0,
        &_ => println!("not supposed to happen"),
    }
    let mut education = 0.0;
    match person.education.as_str() {
        "High School" => education = 0.0,
        "Associate Degree" => education = 1.0,
        "Bachelor's Degree" => education = 2.0,
        "Master's Degree" => education = 3.0,
        "PhD" => education = 4.0,
        &_ => println!("not supposed to happen"),
    }
    let children = person.children as f32;
    let mut smoke = 0.0;
    match person.smoke.as_str() {
        "Non-smoker" => smoke = 0.0,
        "Former" => smoke = 1.0,
        "Current" => smoke = 2.0,
        &_ => println!("not supposed to happen"),
    }
    let mut activity = 0.0;
    match person.activity.as_str() {
        "Sedentary" => activity = 0.0,
        "Moderate" => activity = 1.0,
        "Active" => activity = 2.0,
        &_ => println!("not supposed to happen"),
    }
    let mut employment = 0.0;
    match person.employment {
        true => employment = 1.0,
        false => employment = 0.0,
        // &_ => println!("not supposed to happen"),
    }
    let income = person.income;
    let mut alcohol = 0.0;
    match person.alcohol.as_str() {
        "Low" => alcohol = 0.0,
        "Moderate" => alcohol = 1.0,
        "High" => alcohol = 2.0,
        &_ => println!("not supposed to happen"),
    }
    let mut diet = 0.0;
    match person.diet.as_str() {
        "Unhealthy" => diet = 0.0,
        "Moderate" => diet = 1.0,
        "Healthy" => diet = 2.0,
        &_ => println!("not supposed to happen"),
    }
    let mut sleep = 0.0;
    match person.sleep.as_str() {
        "Poor" => sleep = 0.0,
        "Fair" => sleep = 1.0,
        "Good" => sleep = 2.0,
        &_ => println!("not supposed to happen"),
    }
    let mut substance_abuse = 0.0;
    match person.substance_abuse {
        true => substance_abuse = 1.0,
        false => substance_abuse = 0.0,
        // &_ => println!("not supposed to happen"),
    }
    let mut family_history_depression = 0.0;
    match person.family_history_depression {
        true => family_history_depression = 1.0,
        false => family_history_depression = 0.0,
        // &_ => println!("not supposed to happen"),
    }
    let mut chronic_condition = 0.0;
    match person.chronic_condition {
        true => chronic_condition = 1.0,
        false => chronic_condition = 0.0,
        // &_ => println!("not supposed to happen"),
    }
    let mut mental_illness = 0.0;
    match person.mental_illness {
        true => mental_illness = 1.0,
        false => mental_illness = 0.0,
        // &_ => println!("not supposed to happen"),
    }
    return Individual{name:Person.name, age:age, marital:marital, education:education, children:children, smoke:smoke, activity:activity, employment:employment, income:income, alcohol:alcohol, diet:diet, sleep:sleep, substance_abuse:substance_abuse, family_history_depression:family_history_depression, chronic_condition:chronic_condition, mental_illness:mental_illness};
}

struct Individual {
    name: String,

    age: f32,
    marital: f32,
    education: f32,
    children: f32,
    smoke: f32,
    activity: f32,
    employment: f32,
    income: f32,
    alcohol: f32,
    diet: f32,
    sleep: f32,
    substance_abuse: f32,
    family_history_depression: f32,
    chronic_condition: f32,   

    mental_illness: f32,
}