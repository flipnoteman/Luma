// pub fn extrapolate_dimensions<T>(vec: &Vec<T>) -> Vec<usize> {
//     let mut dimensions = Vec::new();
//     let mut current_level = vec;
//
//     while let Some(first_element) = current_level.first() {
//         dimensions.push(current_level.len());
//         if let Some(next_level) = first_element.downcast_ref::<Vec<T>>() {
//             current_level = next_level;
//         } else {
//             break;
//         }
//     }
//
//     dimensions
// }