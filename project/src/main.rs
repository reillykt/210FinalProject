fn main() {
    let filename = "depression_data_half.csv";
    let people_vector:Vec<Person> = read_csv(filename).unwrap();
    let mut individuals_vector = vec![];
    let mut individuals_vector0 = vec![];
    for person in people_vector {
        individuals_vector.push(to_individual(person.clone()));
        individuals_vector0.push(to_individual(person));
    } 
    let (train_data, test_data) = split(individuals_vector);


    let input_size = 14; // all attributes minus the person's name and the label
    let hidden_size = 5;
    let output_size = 1; // (number between 0 and 1 which I'm going to round to either 0 or 1. Will be used to classify)

    // randomizing weights
    let mut w1: Array2<f32> = Array2::zeros((input_size, hidden_size)); // 14x5 matrix
    populate_array(&mut w1, input_size, hidden_size);
    let mut w2: Array2<f32> = Array2::zeros((hidden_size, output_size)); // 5x1 matrix
    populate_array(&mut w2, hidden_size, output_size);
    

    // creating network
    let mut net = Network{w1, w2};
    let (trained_network, distribution) = train(&mut net, train_data); // training my network and getting back a trained network

    // looks ugly, but I'm just calling forward propagation on vectors where only 1 of the elements is equal to 1 and the rest to 0,
    // and using that to find the attribute that had the biggest impact on calculating the weights for my neural network.
    let (_hidden, age_test) = net.forward_propagation(&specific_element(0));
    let (_hidden, marital_test) = net.forward_propagation(&specific_element(1));
    let (_hidden, education_test) = net.forward_propagation(&specific_element(2));
    let (_hidden, children_test) = net.forward_propagation(&specific_element(3));
    let (_hidden, smoke_test) = net.forward_propagation(&specific_element(4));
    let (_hidden, activity_test) = net.forward_propagation(&specific_element(5));
    let (_hidden, employment_test) = net.forward_propagation(&specific_element(6));
    let (_hidden, income_test) = net.forward_propagation(&specific_element(7));
    let (_hidden, alcohol_test) = net.forward_propagation(&specific_element(8));
    let (_hidden, diet_test) = net.forward_propagation(&specific_element(9));
    let (_hidden, sleep_test) = net.forward_propagation(&specific_element(10));
    let (_hidden, substance_test) = net.forward_propagation(&specific_element(11));
    let (_hidden, fam_hist_test) = net.forward_propagation(&specific_element(12));
    let (_hidden, chronic_test) = net.forward_propagation(&specific_element(13));

    let all_features = vec!["age", "marital status", "education level", "number of children", "smoker status", "physical activity", "employment", "income", "alcohol consumption", "diet", "sleep quality", "substance use/abuse", "family history of depression", "Chronic medical condition"];
    let vec_importance = vec![age_test[(0,0)], marital_test[(0,0)], education_test[(0,0)], children_test[(0,0)], smoke_test[(0,0)], activity_test[(0,0)], employment_test[(0,0)], 
    income_test[(0,0)], alcohol_test[(0,0)], diet_test[(0,0)], sleep_test[(0,0)], substance_test[(0,0)], fam_hist_test[(0,0)], chronic_test[(0,0)]];
    let mut most_important = 0.0;
    let mut important_feature = "?";
    let mut most_important_index = 0;
    for (i,value) in vec_importance.iter().enumerate() {
        if *value > most_important {
            most_important = *value;
            important_feature = all_features[i];
            most_important_index = i;
        }
    }
    println!("\nThe most important value according to the weights calculated by the neural network is an individual's {:?}.", important_feature);

    test(trained_network, test_data, distribution); // using the trained network to test

    hypothesis_test(most_important_index, individuals_vector0);
}

mod read_and_split;
use crate::read_and_split::read_split::*;
use ndarray::Array2;
use rand::Rng;
// use statrs::statistics::Distribution;
// use statrs::distribution::{Normal, Continuous};


fn train(net:&mut Network, train_data:Vec<Individual>) -> (Network,Vec<f32>) {
    let input_size = 14;
    let output_size = 1;

    let mut w1 = net.w1.clone();
    let mut w2 = net.w2.clone();

    let mut all_outputs: Vec<f32> = Vec::new();

    for individual in train_data.into_iter() {
        net.w1 = w1.clone();
        net.w2 = w2.clone();

        let mut input:Array2<f32> = Array2::zeros((1,input_size)); // 1x14 array
        for (index, val) in individual.data().iter().enumerate() {
            input[(0, index)] = *val;
        }
        let (hidden_layer, output) = net.forward_propagation(&input);
        all_outputs.push(output[(0,0)]);

        let (new_w1,new_w2) = net.backward_propagation(&input,&hidden_layer,&output,&individual.mental_illness,output_size);
        w1 = new_w1;
        w2 = new_w2;
    }
    // finding the mean and standard deviation of the outputs, so I can transform my outputs later on to fit a distribution more suited to rounding to 0 or 1
    let output_mean = mean(all_outputs.clone());
    let output_standard_deviation = standard_deviation(output_mean, all_outputs.clone());
    let distribution = vec![output_mean, output_standard_deviation];

    let trained_w1 = net.w1.clone();
    let trained_w2 = net.w2.clone();
    let trained_network = Network{w1: trained_w1, w2: trained_w2};
    
    return (trained_network,distribution);
}

fn test (net:Network, test_data: Vec<Individual>, distribution:Vec<f32>) {
    let input_size = 14;
    let output_size = 1;

    let mut count = 0;
    let mut all_outputs = Vec::new();
    let mut all_targets = Vec::new();

    for individual in test_data {
        count += 1;

        let mut input:Array2<f32> = Array2::zeros((1,input_size)); // input is 1tall, 14 wide
        for (index, val) in individual.data().iter().enumerate() {
            input[(0, index)] = *val;
        }

        let (_hidden_output, output) = net.forward_propagation(&input);
        let target = target(&individual.mental_illness, output_size);

        all_outputs.push(output[(0,0)]);
        all_targets.push(target[(0,0)]);
    }
    
    let new_distribution = vec![distribution[0] / (distribution[0] / 0.5), distribution[1] / (distribution[0] / 0.5)]; // I want the mean to be 0.5 and the standard deviation change to be proportional to that
    
    // I'm transforming all of the outputs to fit my new desired distribution which center around a mean of 0.5, so when I round them to 0, or 1, those who are below the mean are rounded to zero, and the opposite for those above.
    let mut standardized_outputs = Vec::new();
    for output in all_outputs {
        let z_score = (output - distribution[0])/distribution[1]; // z = (x - mu) / sd <-- z-score under the original distribution
        let standardized_output = (z_score * new_distribution[1]) + 0.5; // x = (z * sd) + mu <-- using that^ z-score to find where output falls in new distribution
        standardized_outputs.push(standardized_output);
    }
    // println!("original mean of outputs: {:?}, original standard deviation of outputs: {:?}", distribution[0], distribution[1]);
    // println!("transformed mean of outputs: {:?}, transformed standard deviation of outputs: {:?}", new_distribution[0], new_distribution[1]);

    let mut rounded_outputs = Vec::new();
    for output in &standardized_outputs {
        rounded_outputs.push(output.round());
    }
    let errors = elementwise_subtraction(rounded_outputs.clone(), all_targets.clone()); // rounded_outputs should be either 0 or 1, all_targets should be either 0 or 1
    let mut accuracy = 0;
    for error in errors {
        if error == 0.0 {
            accuracy += 1;
        }
    }
    let percentage_correct = accuracy as f32 / count as f32 * 100.0;
    println!("\nAccuracy = {:?}%", percentage_correct);
}

fn elementwise_subtraction(vec_a: Vec<f32>, vec_b: Vec<f32>) -> Vec<f32> {
    vec_a.into_iter().zip(vec_b).map(|(a, b)| a - b).collect()
}

fn populate_array(arr: &mut Array2<f32>, m:usize, n:usize) {
    let mut rng = rand::thread_rng();
    for i in 0..m {
        for j in 0..n {
            arr[(i, j)] = rng.gen_range(0.0..0.1);
        }
    }
}

struct Network {
    w1:Array2<f32>,
    w2:Array2<f32>,
}


impl Network {
    fn sigmoid(&self, arr: &Array2<f32>) -> Array2<f32> {
        arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn sigmoid_derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        x * &(1.0 - x)
    }
    
    // Function that accepts an input and computes the values for the hidden and output layers
    fn forward_propagation(&self,input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {

        let weighted_input:Array2<f32> = input.dot(&self.w1); // gets weighted input --> 1x14 dot 14x5 => 1x5 
        let hidden_output = self.sigmoid(&weighted_input); // 1x5
        
        let output_input = hidden_output.dot(&self.w2); // 1x5 dot 5x1 => 1x1
        let output = output_input.clone(); // 1x1 
        return (hidden_output, output);
    }

    //Function that accepts the input, hidden, output and label information and updates the hidden and output weights
    fn backward_propagation(&mut self, input: &Array2<f32>, hidden_output: &Array2<f32>, output: &Array2<f32>, label: &f32, output_size:usize) -> (Array2<f32>,Array2<f32>) {

        let target = target(label,output_size); // 1x1

        let error = target - output; //  1x1 minus 1x1 => 1x1
        let delta = &error * &self.sigmoid_derivative(&output); // 1x1

        let error_hidden = delta.dot(&self.w2.t()); // 1x1 dot 1x5 => 1x5
        let delta_hidden = &error_hidden * &self.sigmoid_derivative(&hidden_output); // 1x5

        let weight2_updates = hidden_output.t().dot(&delta); // 5x1 dot 1x1 -> 5x1
        self.w2 += &(weight2_updates * 0.10); 

        let weight1_updates = input.t().dot(&delta_hidden); // 14x1 dot 1x5 => 14x5
        self.w1 += &(weight1_updates * 0.10);

        let new_w1 = self.w1.clone();
        let new_w2 = self.w2.clone();
        return (new_w1, new_w2);
    }

}

fn target(label: &f32, output_size: usize) -> Array2<f32> {
    let mut target:Array2<f32> = Array2::zeros((1, output_size)); // 1x1
    if *label == 0.0 {
        target[(0,0)] = 0.0;
    }
    if *label == 1.0 {
        target[(0,0)] = 1.0;
    }
    return target;
}

fn specific_element(index:usize) -> Array2<f32> {
    let mut array:Array2<f32> = Array2::zeros((1, 14));
    array[(0,index)] = 1.0;
    return array
}


fn hypothesis_test(index:usize, individuals_vector:Vec<Individual>) {
    let all_features = vec!["age", "marital status", "education level", "number of children", "smoker status", "physical activity", "employment", "income", "alcohol consumption", "diet", "sleep quality", "substance use/abuse", "family history of depression", "Chronic medical condition"];
    let most_important_feature = all_features[index];

    println!("\nBecause the results of the Neural Network aren't very high, I want to test whether or not the most important attribute, 
    according to the weights calculated in the training of the neural network, actually has a relationship with having a history of mental illness.");
    println!("\nConducting a hypothesis to see if the attribute {:?} actually has a relationship with having a history of mental illness", most_important_feature);
    
    println!("\nNull H0: P(Mental Illness | {:?}) - P(Mental Illness) = 0", most_important_feature);
    println!("Alternate HA: P(Mental Illness | {:?}) - P(Mental Illness) > 0", most_important_feature);

    let mut mentally_ill = 0.0;
    let mut mentally_ill_with_condition = 0.0;
    let mut total_with_chronic = 0.0;
    let mut total_individuals = 0.0;
    for individual in individuals_vector {
        total_individuals += 1.0;
        if individual.mental_illness == 1.0 {
            mentally_ill += 1.0;
        }
        if individual.chronic_condition == 1.0 {
            total_with_chronic += 1.0;
            if individual.mental_illness == 1.0 {
                mentally_ill_with_condition += 1.0;
            }
        }
    }
    let p_hat = mentally_ill_with_condition/total_with_chronic - mentally_ill/total_individuals;
    println!("\nthe observed statistic is: {:?}", p_hat);
}

fn standard_deviation(mean:f32, all_datapoints: Vec<f32>) -> f32 {
    let mut count = 0.0;
    let mut squared_differences = 0.0;
    for datapoint in all_datapoints {
        count += 1.0;
        squared_differences += (datapoint - mean) * (datapoint - mean);
    }
    let variance = squared_differences / count;
    let standard_deviation = variance.sqrt();
    return standard_deviation;
}

fn mean(all_datapoints: Vec<f32>) -> f32 {
    let mut total = 0.0;
    let mut count = 0.0;
    for output in all_datapoints {
        total += output;
        count += 1.0;
    }
    let mean = total / count;
    return mean;
}