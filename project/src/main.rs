fn main() {
    let filename = "depression_data_half.csv";
    let people_vector:Vec<Person> = read_csv(filename).unwrap();
    let mut individuals_vector = vec![];
    for person in people_vector {
        individuals_vector.push(to_individual(person));
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
    let (trained_network,min_and_max) = train(&mut net, train_data); // training my network and getting back a trained network

    // let (hidden, age_test) = net.forward_propagation(&specific_element(0));
    // let (hidden, marital_test) = net.forward_propagation(&specific_element(1));
    // let (hidden, education_test) = net.forward_propagation(&specific_element(2));
    // let (hidden, children_test) = net.forward_propagation(&specific_element(3));
    // let (hidden, smoke_test) = net.forward_propagation(&specific_element(4));
    // let (hidden, activity_test) = net.forward_propagation(&specific_element(5));
    // let (hidden, employment_test) = net.forward_propagation(&specific_element(6));
    // let (hidden, income_test) = net.forward_propagation(&specific_element(7));
    // let (hidden, alcohol_test) = net.forward_propagation(&specific_element(8));
    // let (hidden, diet_test) = net.forward_propagation(&specific_element(9));
    // let (hidden, sleep_test) = net.forward_propagation(&specific_element(10));
    // let (hidden, substance_test) = net.forward_propagation(&specific_element(11));
    // let (hidden, fam_hist_test) = net.forward_propagation(&specific_element(12));
    // let (hidden, chronic_test) = net.forward_propagation(&specific_element(13));

    // println!("age: {:?}, marital: {:?}, education: {:?}, children: {:?}, smoke: {:?}, activity: {:?}, employment: {:?},
    //  income: {:?}, alcohol: {:?}, diet: {:?}, sleep: {:?}, substance abuse: {:?}, family history: {:?}, chronic condition: {:?}", 
    // age_test[(0,0)], marital_test[(0,0)], education_test[(0,0)], children_test[(0,0)], smoke_test[(0,0)], activity_test[(0,0)], employment_test[(0,0)], 
    // income_test[(0,0)], alcohol_test[(0,0)], diet_test[(0,0)], sleep_test[(0,0)], substance_test[(0,0)], fam_hist_test[(0,0)], chronic_test[(0,0)]);


    test(trained_network, test_data, min_and_max); // using the trained network to test
}

mod read_and_split;
use crate::read_and_split::read_split::*;
use ndarray::Array2;
use rand::Rng;


fn train(net:&mut Network, train_data:Vec<Individual>) -> (Network,Vec<f32>) {
    let input_size = 14;
    let output_size = 1;

    let mut w1 = net.w1.clone();
    let mut w2 = net.w2.clone();

    let mut min_and_max = vec![0.2, 0.2];
    // let mut ill_outputs = Vec::new();
    // let mut fine_outputs = Vec::new();
    for individual in train_data.into_iter() {
        // if indices.contains(&in_dex) { 
        net.w1 = w1.clone();
        net.w2 = w2.clone();

        let mut input:Array2<f32> = Array2::zeros((1,input_size)); // 1x14 array
        for (index, val) in individual.data().iter().enumerate() {
            input[(0, index)] = *val;
        }
        let (hidden_layer, output) = net.forward_propagation(&input);
        if output[(0,0)] < min_and_max[0] {
            min_and_max[0] = output[(0,0)];
            println!("new minimum: {:?}", min_and_max[0]);
        }
        if output[(0,0)] > min_and_max[1] {
            min_and_max[1] = output[(0,0)];
            println!("new maximum: {:?}", min_and_max[1]);
        }
        // let target = target(&individual.mental_illness, output_size);
        // if target[(0,0)] == 1.0 {
        //     ill_outputs.push(output[(0,0)]);
        // }
        // if target[(0,0)] == 0.0 {
        //     fine_outputs.push(output[(0,0)]);
        // }
        let (new_w1,new_w2) = net.backward_propagation(&input,&hidden_layer,&output,&individual.mental_illness,output_size);
        w1 = new_w1;
        w2 = new_w2;
    }
    // let mut ill_sum = 0.0;
    // for number in ill_outputs.clone().into_iter() {
    //     ill_sum += number;
    // }
    // let mut fine_sum = 0.0;
    // for number in fine_outputs.clone().into_iter() {
    //     fine_sum += number;
    // }
    // let ill_average = ill_sum / (ill_outputs.clone().into_iter().len() as f32);
    // let fine_average = fine_sum / (fine_outputs.clone().into_iter().len() as f32);
    // println!("mean ill output: {:?}, mean fine output: {:?}",ill_average, fine_average);
    // println!("final ill output: {:?}, final fine output: {:?}",ill_outputs.pop(), fine_outputs.pop());
    // println!("2nd final ill output: {:?}, 2nd final fine output: {:?}",ill_outputs.pop(), fine_outputs.pop());
    // println!("3rd final ill output: {:?}, 3rd final fine output: {:?}",ill_outputs.pop(), fine_outputs.pop());
    let trained_w1 = net.w1.clone();
    let trained_w2 = net.w2.clone();
    let trained_network = Network{w1: trained_w1, w2: trained_w2};
    println!("min/max: {:?}", min_and_max);
    return (trained_network, min_and_max);
}

fn test (net:Network, test_data: Vec<Individual>, min_and_max:Vec<f32>) {
    let input_size = 14;
    // let hidden_size = 5;
    let output_size = 1;

    let mut accuracy = 0.0;
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


        // let mut error = output[(0,0)].round() - target[(0,0)]; // should be either 0 or 1
        // if error == 0.0 {
        //     accuracy += 1.0;
        // }
        // println!("output: {:?}, target: {:?}", output[(0,0)], target[(0,0)]);
    }
    println!("{:?}", all_outputs[0]);
    let mut y = elementwise_subtraction(all_outputs, vec![min_and_max[0]; (count.clone() as usize)]);
    println!("{:?}", y[0]);
    println!("max - min {:?}", (min_and_max[1] - min_and_max[0]));
    // println!("min/max: {:?}", min_and_max);
    let x = 1.0 / (min_and_max[1] - min_and_max[0]);
    println!("{:?}", x);
    for i in 0..count {
        y[i] *= x;
        y[i] = y[i].round();
        // println!("{:?}", y[i]);
    }
    let errors = elementwise_subtraction(y, all_targets);
    let mut accuracy = 0;
    for error in errors {
        if error == 0.0 {
            accuracy += 1;
        }
    }
    println!("\nAccuracy = {:?}%",accuracy/count * 100);
    println!("currently says that every output is equal to zero .... and the proportion of individuals without mental illness is 69%")
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

        // println!("input: {:?}", input);
        // println!("w1: {:?}", self.w1);
        let weighted_input:Array2<f32> = input.dot(&self.w1); // gets weighted input --> 1x14 dot 14x5 => 1x5 
        // println!("weighted_input: {:?}", weighted_input);
        let hidden_output = self.sigmoid(&weighted_input); // 1x5
        // println!("hidden_output: {:?}", hidden_output);
        
        let output_input = hidden_output.dot(&self.w2); // 1x5 dot 5x1 => 1x1
        // println!("output_input: {:?}", output_input);
        let output = output_input.clone(); // 1x1 
        // println!("output: {:?}\n", output);
        return (hidden_output, output);
    }

    //Function that accepts the input, hidden, output and label information and updates the hidden and output weights
    fn backward_propagation(&mut self, input: &Array2<f32>, hidden_output: &Array2<f32>, output: &Array2<f32>, label: &f32, output_size:usize) -> (Array2<f32>,Array2<f32>) {

        let target = target(label,output_size); // 1x1

        let error = target - output; //  1x1 minus 1x1 => 1x1
        // println!("error: {:?}", error);
        // let delta = &error * &self.sigmoid_derivative(&output); // 1x1
        let delta = &error * output; // trying things

        let error_hidden = delta.dot(&self.w2.t()); // 1x1 dot 1x5 => 1x5
        // let delta_hidden = &error_hidden * &self.sigmoid_derivative(&hidden_output); // 1x5
        let delta_hidden = &error_hidden * hidden_output; // trying things

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


// fn hypothesis_test {
//     println!("Because the test results ")
//     println!("Conducting a hypothesis to see if the seemingly most significant attribute (whether or not the individual has a chronic condition)
//     actually has a relationship with having a history of mental illness");
//     println!("Null H0: P(Mental Illness | Chronic Condition) - P(Mental Illness) = 0")
//     println!("Alternate HA: P(Mental Illness | Chronic Condition) - P(Mental Illness) > 1")

//     let mut mentally_ill = 0;
//     let mut mentally_ill_with_condition = 0;
//     let mut total_with_chronic = 0;
//     let mut total_individuals = 0;
//     for individual in individuals_vector {
//         total_individuals += 1;
//         if individual.mental_illness == 1.0 {
//             mentally_ill += 1;
//         }
//         if individual.chronic_condition == 1.0 {
//             total_with_chronic += 1;
//             if individual.mental_illness == 1.0 {
//                 mentally_ill_with_condition += 1;
//             }
//         }
//     }
//     let p_hat = mentally_ill_with_condition/total_with_chronic - mentally_ill/total_individuals;

// }