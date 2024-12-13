fn main() {
    let filename = "depression_data_half.csv";
    let people_vector:Vec<Person> = read_csv(filename).unwrap();
    let mut individuals_vector = vec![];
    for person in people_vector {
        individuals_vector.push(to_individual(person));
    }
    let (train_data, test_data) = split(individuals_vector);
}

mod read_and_split;
use crate::read_and_split::read_split::*;
