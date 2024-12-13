pub mod read_split {

    // use serde::Deserialize;
    use std::path::Path;
    use std::fs;
    use csv;
    use std::error::Error;

    #[derive(Debug)]
    pub struct Person {
        pub name: String,
        pub age: String,
        pub marital: String,
        pub education: String,
        pub children: String,
        pub smoke: String,
        pub activity: String,
        pub employment: String,
        pub income: String,
        pub alcohol: String,
        pub diet: String,
        pub sleep: String,
        pub mental_illness: String,
        pub substance_abuse: String,
        pub family_history_depression: String,
        pub chronic_condition: String,  
    }

    #[derive(Debug)]
    pub struct Individual {
        pub name: String,

        pub age: f32,
        pub marital: f32,
        pub education: f32,
        pub children: f32,
        pub smoke: f32,
        pub activity: f32,
        pub employment: f32,
        pub income: f32,
        pub alcohol: f32,
        pub diet: f32,
        pub sleep: f32,
        pub substance_abuse: f32,
        pub family_history_depression: f32,
        pub chronic_condition: f32,   

        pub mental_illness: f32,
    }

    pub fn read_csv<P: AsRef<Path> + std::fmt::Display>(filename: P) -> Result<Vec<Person>, Box<dyn Error>> {
        // Check if the file exists before trying to open it
        if fs::metadata(&filename).is_err() {
            eprintln!("The file '{}' does not exist or cannot be accessed.", filename);
            return Err("File not found".into());
        }
        let mut rdr = csv::Reader::from_path(filename)?;
        let mut vector_of_people:Vec<Person> = Vec::new();
    
    
        for (index,row) in rdr.deserialize::<Vec<String>>().enumerate() { // for Vec<String> in rdr
            if index != 0 {
                let unwrapped = row.unwrap();//.split(","); // 
                // let mut unwrapped = vec![];
                // for elem in unwrapped0 {
                //     unwrapped.push(elem); // unwrapped is Vec<&str>
                // }
                let name = unwrapped[0].to_string();
                let age = unwrapped[1].to_string();
                let marital = unwrapped[2].to_string();
                let education = unwrapped[3].to_string();
                let children = unwrapped[4].to_string();
                let smoke = unwrapped[5].to_string();
                let activity = unwrapped[6].to_string();
                let employment = unwrapped[7].to_string();
                let income = unwrapped[8].to_string();
                let alcohol = unwrapped[9].to_string();
                let diet = unwrapped[10].to_string();
                let sleep = unwrapped[11].to_string();
                let mental_illness = unwrapped[12].to_string();
                let substance_abuse = unwrapped[13].to_string();
                let family_history_depression = unwrapped[14].to_string();
                let chronic_condition = unwrapped[15].to_string();
    
                // let mut mental_illness = false;
                // if unwrapped[11] == "Yes".to_string() {
                //     mental_illness = true;
                // }
                let person = Person{name:name, age:age, marital:marital, education:education, children:children, smoke:smoke, activity:activity, employment:employment, income:income, alcohol:alcohol, diet:diet, sleep:sleep, mental_illness:mental_illness, substance_abuse:substance_abuse, family_history_depression:family_history_depression, chronic_condition:chronic_condition};
                vector_of_people.push(person);
            }
        }
        return Ok(vector_of_people);
    }

    pub fn to_individual(person:Person) -> Individual {
        let age = person.age.parse::<f32>();
        let mut marital = 0.0;
        // println!("{:?}", person.marital);
        match person.marital.as_str() {
            "Single" => marital = 1.0,
            "Married" => marital = 2.0,
            "Divorced" => marital = 3.0,
            "Widowed" => marital = 4.0,
            &_ => println!("not supposed to happen"),
        }
        let mut education = 0.0;
        // println!("{:?}", person.education);
        match person.education.as_str() {
            "High School" => education = 0.0,
            "Associate Degree" => education = 1.0,
            "Bachelor's Degree" => education = 2.0,
            "Master's Degree" => education = 3.0,
            "PhD" => education = 4.0,
            &_ => println!("not supposed to happen"),
        }
        let children = person.children.parse::<f32>();
        let mut smoke = 0.0;
        // println!("{:?}", person.smoke);
        match person.smoke.as_str() {
            "Non-smoker" => smoke = 0.0,
            "Former" => smoke = 1.0,
            "Current" => smoke = 2.0,
            &_ => println!("not supposed to happen"),
        }
        let mut activity = 0.0;
        // println!("{:?}", person.activity);
        match person.activity.as_str() {
            "Sedentary" => activity = 0.0,
            "Moderate" => activity = 1.0,
            "Active" => activity = 2.0,
            &_ => println!("not supposed to happen"),
        }
        let mut employment = 0.0;
        // println!("{:?}", person.employment);
        match person.employment.as_str() {
            "Employed" => employment = 1.0,
            "Unemployed" => employment = 0.0,
            &_ => println!("not supposed to happen"),
        }
        let income = person.income.parse::<f32>();
        let mut alcohol = 0.0;
        // println!("{:?}", person.alcohol);
        match person.alcohol.as_str() {
            "Low" => alcohol = 0.0,
            "Moderate" => alcohol = 1.0,
            "High" => alcohol = 2.0,
            &_ => println!("not supposed to happen"),
        }
        let mut diet = 0.0;
        // println!("{:?}", person.diet);
        match person.diet.as_str() {
            "Unhealthy" => diet = 0.0,
            "Moderate" => diet = 1.0,
            "Healthy" => diet = 2.0,
            &_ => println!("not supposed to happen"),
        }
        let mut sleep = 0.0;
        // println!("{:?}", person.sleep);
        match person.sleep.as_str() {
            "Poor" => sleep = 0.0,
            "Fair" => sleep = 1.0,
            "Good" => sleep = 2.0,
            &_ => println!("not supposed to happen"),
        }
        let mut substance_abuse = 0.0;
        // println!("{:?}", person.substance_abuse);
        match person.substance_abuse.as_str() {
            "Yes" => substance_abuse = 1.0,
            "No" => substance_abuse = 0.0,
            &_ => println!("not supposed to happen"),
        }
        let mut family_history_depression = 0.0;
        // println!("{:?}", person.family_history_depression);
        match person.family_history_depression.as_str() {
            "Yes" => family_history_depression = 1.0,
            "No" => family_history_depression = 0.0,
            &_ => println!("not supposed to happen"),
        }
        let mut chronic_condition = 0.0;
        match person.chronic_condition.as_str() {
            "Yes" => chronic_condition = 1.0,
            "No" => chronic_condition = 0.0,
            &_ => println!("not supposed to happen"),
        }
        let mut mental_illness = 0.0;
        match person.mental_illness.as_str() {
            "Yes" => mental_illness = 1.0,
            "No" => mental_illness = 0.0,
            &_ => println!("not supposed to happen"),
        }
        let name = person.name;
        return Individual{name:name, age:age.unwrap(), marital:marital, education:education, children:children.unwrap(), smoke:smoke, activity:activity, employment:employment, income:income.unwrap(), alcohol:alcohol, diet:diet, sleep:sleep, substance_abuse:substance_abuse, family_history_depression:family_history_depression, chronic_condition:chronic_condition, mental_illness:mental_illness};
    }

    pub fn split(all_data: Vec<Individual>) -> (Vec<Individual>, Vec<Individual>) {
        let mut train_data:Vec<Individual> = vec![];
        let mut test_data:Vec<Individual> = vec![];
        for (index,row) in all_data.into_iter().enumerate() {
            if index < 145000 {
                train_data.push(row);
            }
            else {
                test_data.push(row);
            }
        }
        return (train_data, test_data)
    }
}


// pub mod split {

//     // pub struct Individual {
//     //     pub name: String,

//     //     pub age: f32,
//     //     pub marital: f32,
//     //     pub education: f32,
//     //     pub children: f32,
//     //     pub smoke: f32,
//     //     pub activity: f32,
//     //     pub employment: f32,
//     //     pub income: f32,
//     //     pub alcohol: f32,
//     //     pub diet: f32,
//     //     pub sleep: f32,
//     //     pub substance_abuse: f32,
//     //     pub family_history_depression: f32,
//     //     pub chronic_condition: f32,   

//     //     pub mental_illness: f32,
//     // }

//     pub fn split(all_data: Vec<Individual>) -> (Vec<Individual>, Vec<Individual>) {
//         let mut train_data:Vec<Individual> = vec![];
//         let mut test_data:Vec<Individual> = vec![];
//         for (index,row) in all_data.into_iter().enumerate() {
//             if index < 145000 {
//                 train_data.push(row);
//             }
//             else {
//                 test_data.push(row);
//             }
//         }
//         return (train_data, test_data)
//     }
// }