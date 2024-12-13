fn main() {
    let filename = "depression_data_half.csv";
    let people_vector:Vec<Person> = read_csv(filename).unwrap();
    let mut individuals_vector = vec![];
    for person in people_vector {
        individuals_vector.push(to_individual(person));
    }
    let (train_data, test_data) = split(individuals_vector);


    let input_size = 15; // all attributes minus the person's name
    let hidden_size = 5;
    let output_size = 1; // (number between 0 and 1 which I'm going to round to either 0 or 1. Will be used to classify)

    // randomizing weights
    let mut w1: Array2<f32> = Array2::zeros((input_size, hidden_size)); // 15x5 matrix (basically a single vector)
    populate_array(&mut w1, input_size, hidden_size);
    let mut w2: Array2<f32> = Array2::zeros((hidden_size, output_size)); // 5x1 matrix (basically a single vector)
    populate_array(&mut w2, hidden_size, output_size);
    

    // creating network
    let mut net = Network{w1, w2};
    let trained_network = train(&mut net, train_data); // training my network and getting back a trained network
    test(trained_network, test_data); // using the trained network to test
}

mod read_and_split;
use crate::read_and_split::read_split::*;
use ndarray::Array2;
use rand::Rng;


fn train(net:&mut Network, train_data:Vec<Individual>) -> Network {
    let input_size = 15;
    let output_size = 1;

    let mut w1 = net.w1.clone();
    let mut w2 = net.w2.clone();

    for individual in train_data {
        net.w1 = w1.clone();
        net.w2 = w2.clone();

        let mut input:Array2<f32> = Array2::zeros((1,input_size)); 
        for (index, val) in individual.data().iter().enumerate() {
            input[(0, index)] = *val;
        }
        let (hidden_layer, output) = net.forward_propagation(&input);
        let (new_w1,new_w2) = net.backward_propagation(&input,&hidden_layer,&output,&individual.mental_illness,output_size);
        w1 = new_w1;
        w2 = new_w2;
    }
    let trained_w1 = net.w1.clone();
    let trained_w2 = net.w2.clone();
    let trained_network = Network{w1: trained_w1, w2: trained_w2};
    return trained_network;
}

fn test (net:Network, test_data: Vec<Individual>) {
    let input_size = 15;
    // let hidden_size = 5;
    let output_size = 1;

    let mut accuracy = 0.0;
    let mut count = 0.0;
    for individual in test_data {
        count += 1.0;
        let mut input:Array2<f32> = Array2::zeros((1,input_size)); // input is 1tall, 15 wide
        for (index, val) in individual.data().iter().enumerate() {
            input[(0, index)] = *val;
        }
        let (_hidden_output, output) = net.forward_propagation(&input);
        let target = target(&individual.mental_illness, output_size);

        let mut error = output[(0,0)].round() - target[(0,0)]; // should be either 0 or 1
        if error == 0.0 {
            accuracy += 1.0;
        }
        // println!("output: {:?}, target: {:?}", output[(0,0)].round(), target[(0,0)]);
    }
    println!("\nAccuracy = {:?}%",accuracy/count * 100.0);
}


fn populate_array(arr: &mut Array2<f32>, m:usize, n:usize) {
    let mut rng = rand::thread_rng();
    for i in 0..m {
        for j in 0..n {
            arr[(i, j)] = rng.gen_range(0.0..1.0);
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

        let weighted_input:Array2<f32> = input.dot(&self.w1); // gets weighted input --> 1x15 dot 15x5 => 1x5 
        let hidden_output = self.sigmoid(&weighted_input); // 1x5

        let output_input = hidden_output.dot(&self.w2); // 1x5 dot 5x1 => 1x1
        let output = self.sigmoid(&output_input); // 1x1 

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

        let weight1_updates = input.t().dot(&delta_hidden); // 15x1 dot 1x5 => 15x5
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
