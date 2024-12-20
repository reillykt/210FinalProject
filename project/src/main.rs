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

    test(trained_network, test_data, distribution); // using the trained network to test

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

    let vec_importance = vec![age_test[(0,0)], marital_test[(0,0)], education_test[(0,0)], children_test[(0,0)], smoke_test[(0,0)], activity_test[(0,0)], employment_test[(0,0)], 
    income_test[(0,0)], alcohol_test[(0,0)], diet_test[(0,0)], sleep_test[(0,0)], substance_test[(0,0)], fam_hist_test[(0,0)], chronic_test[(0,0)]];
    let mut most_important = 0.0;
    let mut most_important_index = 0;
    for (i,value) in vec_importance.iter().enumerate() {
        if *value > most_important {
            most_important = *value;
            most_important_index = i;
        }
    }

    hypothesis_test(most_important_index, individuals_vector0);
}

mod read_and_split;
use crate::read_and_split::read_split::*;
use ndarray::Array2;
use rand::Rng;
use std::thread;
use std::time::Duration;


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
    println!("initial data distribution: {:?}", distribution);

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
    println!("scaled data distribution: {:?}", new_distribution);

    // I'm transforming all of the outputs to fit my new desired distribution which center around a mean of 0.5, so when I round them to 0, or 1, those who are below the mean are rounded to zero, and the opposite for those above.
    let mut standardized_outputs = Vec::new();
    for output in &all_outputs {
        let z_score = (output - distribution[0])/distribution[1]; // z = (x - mu) / sd <-- z-score under the original distribution
        let standardized_output = (z_score * new_distribution[1]) + 0.5; // x = (z * sd) + mu <-- using that^ z-score to find where output falls in new distribution
        standardized_outputs.push(standardized_output);
    }
    // println!("original mean of outputs: {:?}, original standard deviation of outputs: {:?}", distribution[0], distribution[1]);
    // println!("transformed mean of outputs: {:?}, transformed standard deviation of outputs: {:?}", new_distribution[0], new_distribution[1]);

    let mut rounded_original = Vec::new();
    for output in &all_outputs {
        rounded_original.push(output.round());
    }

    let mut rounded_outputs = Vec::new();
    for output in &standardized_outputs {
        rounded_outputs.push(output.round());
    }
    let original_errors = elementwise_subtraction(rounded_original.clone(), all_targets.clone()); // rounded_outputs should be either 0 or 1, all_targets should be either 0 or 1
    let errors = elementwise_subtraction(rounded_outputs.clone(), all_targets.clone()); // rounded_outputs should be either 0 or 1, all_targets should be either 0 or 1

    let mut original_accuracy = 0;
    for error in original_errors {
        if error == 0.0 {
            original_accuracy += 1;
        }
    }
    let mut accuracy = 0;
    for error in errors {
        if error == 0.0 {
            accuracy += 1;
        }
    }
    let percentage_correct = accuracy as f32 / count as f32 * 100.0;
    let original_percentage_correct = original_accuracy as f32 / count as f32 * 100.0;
    println!("\nUsing the original distribution, the accuracy of the neural network to indentify mental illness within an individual is {:?}%", original_percentage_correct);
    println!("\nThe accuracy of the neural network to indentify mental illness within an individual is {:?}%", percentage_correct);
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
        self.w2 += &(weight2_updates * 0.1); 

        let weight1_updates = input.t().dot(&delta_hidden); // 14x1 dot 1x5 => 14x5
        self.w1 += &(weight1_updates * 0.1);

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

    thread::sleep(Duration::new(1, 0));
    println!("\nBecause the accuracy of the Neural Network wasn't very high, I want to test whether or not the most important attribute, according to the weights calculated in the training of the neural network, actually has a relationship with having a history of mental illness.");
    println!("\nConducting a 2-sample independent z-test to see if the attribute {:?} actually has a relationship with having a history of mental illness", most_important_feature);
    
    thread::sleep(Duration::new(1, 0));

    println!("\nNull H0: P(Mental Illness | {:?}) - P(Mental Illness) = 0", most_important_feature); 
    // same as P(Mentally ill|Has Chronic) - P(Mentally ill| Does Not Have Chronic)
    println!("Alternate HA: P(Mental Illness | {:?}) - P(Mental Illness) > 0", most_important_feature);

    let mut mentally_ill_with_condition = 0.0;
    let mut total_with_chronic = 0.0;

    let mut mentally_ill_without_condition = 0.0;
    let mut total_without_chronic = 0.0;

    for individual in individuals_vector {
        if individual.chronic_condition == 1.0 {
            total_with_chronic += 1.0;
            if individual.mental_illness == 1.0 {
                mentally_ill_with_condition += 1.0;
            }
        }
        if individual.chronic_condition == 0.0 {
            total_without_chronic += 1.0;
            if individual.mental_illness == 1.0 {
                mentally_ill_without_condition += 1.0;
            }
        }
    }
    let p_hat1 = mentally_ill_with_condition/total_with_chronic;
    let p_hat2 = mentally_ill_without_condition/total_without_chronic;
    let p_hat = (mentally_ill_with_condition + mentally_ill_without_condition) as f32 /(total_with_chronic + total_without_chronic) as f32;
    let q_hat = 1.0 as f32 - p_hat;
    let standard_error = (p_hat * q_hat * ((1.0/total_with_chronic) + (1.0/total_without_chronic)) as f32).sqrt();
    let test_statistic = (p_hat1 - p_hat2) / standard_error; // test statistic is Z
    let rejection_region = 1.281552;

    thread::sleep(Duration::new(1, 0));
    println!("The test statistic Z = {:?}", test_statistic);
    println!("The rejection region for the test statistic is anything where Z > {:?}", rejection_region);

    thread::sleep(Duration::new(1, 0));
    println!("\nThe hypothesis test provided us with enough evidence to reject the null hypothesis that having a chronic medical condition does not impact having a history with mental illness.");
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_propagation() {
        let input_size = 3;
        let hidden_size = 2;
        let output_size = 1;
    
        let mut input:Array2<f32> = Array2::zeros((1, 3));
        input[(0,0)] = 1.0 as f32;
        input[(0,1)] = 2.0 as f32;
        input[(0,2)] = 3.0 as f32;

        // randomizing weights
        let mut w1: Array2<f32> = Array2::zeros((input_size, hidden_size));
        populate_array(&mut w1, input_size, hidden_size);
        let mut w2: Array2<f32> = Array2::zeros((hidden_size, output_size));
        populate_array(&mut w2, hidden_size, output_size);

        let net = Network{w1, w2};

        let (_hidden_layer, output) = net.forward_propagation(&input);

        // Check if the output has the correct size (should be 1 since there is 1 output)
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn testing_weights_initialization() {
        let input_size = 3;
        let hidden_size = 2;
        let output_size = 1;

        // randomizing weights
        let mut w1: Array2<f32> = Array2::zeros((input_size, hidden_size));
        populate_array(&mut w1, input_size, hidden_size);
        let mut w2: Array2<f32> = Array2::zeros((hidden_size, output_size));
        populate_array(&mut w2, hidden_size, output_size);

        let net = Network{w1, w2};

        // Checking dimensions of the weight matrices
        assert_eq!(net.w1.shape(), &[3, 2]);  // 3 input, 2 hidden
        assert_eq!(net.w2.shape(), &[2, 1]);  // 2 hidden, 1 output
    }

    #[test]
    fn testing_sigmoid_activation() {
        let input_size = 3;
        let hidden_size = 2;
        let output_size = 1;

        // randomizing weights
        let mut w1: Array2<f32> = Array2::zeros((input_size, hidden_size));
        populate_array(&mut w1, input_size, hidden_size);
        let mut w2: Array2<f32> = Array2::zeros((hidden_size, output_size));
        populate_array(&mut w2, hidden_size, output_size);

        let net = Network{w1, w2};

        let mut values:Array2<f32> = Array2::zeros((1, 5));
        values[(0,0)] = -100.0 as f32;
        values[(0,1)] = -10.0 as f32;
        values[(0,2)] = 0.0 as f32;
        values[(0,3)] = 10.0 as f32;
        values[(0,4)] = 100.0 as f32;

        let values_vector = vec![-100.0, -10.0, 0.0, 10.0, 100.0];
        
        let results = net.sigmoid(&values);
        let mut results_vector = Vec::new();
        for i in 1..5 {
            results_vector.push(results[(0,i)]);
        }
        for (i,result) in results_vector.iter().enumerate() {
            assert!(result >= &(0.0 as f32) && result <= &(1.0 as f32), "Sigmoid({}) should be in the range [0, 1]", values_vector[i]);
        }
    }
}