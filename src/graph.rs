//! Implementation of some functions related to graph theory necessary for setting up the methods.

/// [NodeState] represents the state of a node in gabow's algorithm.
#[derive(Debug, Clone, PartialEq)]
enum NodeState {
    NotVisited,
    InPath(usize),
    InSCC(usize),
}

/// [SCC()] is an implementation of Gabow's algorithm, an algorithm which detects the strongly connected components of a digraph.
pub fn SCC(graph: &Vec<Vec<bool>>) -> Vec<Vec<usize>> {
    #[allow(non_snake_case)]
    let mut S: Vec<usize> = Vec::new();
    #[allow(non_snake_case)]
    let mut B: Vec<usize> = Vec::new();
    let s = graph.len();
    if s == 0 {
        return Vec::new();
    }
    let mut scc_count: usize = 0;
    let mut state: Vec<NodeState> = vec![NodeState::NotVisited; s];
    let mut tree: Vec<Vec<usize>> = vec![(0..s).collect()];
    let mut l: usize;
    let mut path: Vec<usize> = Vec::new();
    let mut finished = false;
    while !finished {
        l = tree.len() - 1;
        match tree[l].pop() {
            Some(j) => {
                // next node to search
                match state[j] {
                    NodeState::NotVisited => {
                        path.push(j);
                        S.push(j);
                        state[j] = NodeState::InPath(S.len() - 1);
                        B.push(S.len() - 1);
                        tree.push((0..s).filter(|&j1| graph[j][j1]).collect());
                    }
                    NodeState::InPath(j1) => {
                        while j1 < B[B.len() - 1] {
                            B.pop();
                        }
                    }
                    _ => {}
                }
            }
            None => {
                // all edges of current node were searched
                tree.pop();
                match path.pop() {
                    Some(j) => {
                        // assign to the scc
                        if let NodeState::InPath(j1) = state[j] {
                            if j1 == B[B.len() - 1] {
                                B.pop();
                                while j1 < S.len() {
                                    state[S.pop().unwrap()] = NodeState::InSCC(scc_count);
                                }
                                scc_count += 1;
                            }
                        }
                    }
                    None => finished = true, // no more path to search
                }
            }
        }
    }
    let mut scc = vec![Vec::new(); scc_count];
    for j in 0..s {
        if let NodeState::InSCC(scc_index) = state[j] {
            scc[scc_index].push(j);
        }
    }
    scc
}

/// [contraction] takes a digraph and a partition of the nodes and outpus the contracted graph, without the loops.
pub fn contraction(graph: &Vec<Vec<bool>>, partition: &Vec<Vec<usize>>) -> Vec<Vec<bool>> {
    let s = graph.len();
    let p = partition.len();
    let mut inverse_map: Vec<usize> = vec![0; s];
    for i in 0..p {
        for &j in &partition[i] {
            inverse_map[j] = i;
        }
    }
    let mut contracted_graph = vec![vec![false; p]; p];
    let mut partition1;
    let mut partition2;
    for j1 in 0..s {
        for j2 in 0..s {
            partition1 = inverse_map[j1];
            partition2 = inverse_map[j2];
            if partition1 != partition2 && graph[j1][j2] {
                contracted_graph[partition1][partition2] = true;
            }
        }
    }
    contracted_graph
}

/// [topological_sort] takes a digraph with no non-empty walk and outputs a relabeling which topologically sorts the digraph.
pub fn topological_sort(dag: &Vec<Vec<bool>>) -> Vec<usize> {
    let mut dag = dag.clone();
    let s = dag.len();
    #[allow(non_snake_case)]
    let mut S: Vec<usize> = (0..s).filter(|&j| !(dag[j].contains(&true))).collect();
    let mut order: Vec<usize> = Vec::new();
    while !S.is_empty() {
        for node in S {
            order.push(node);
            for j in 0..s {
                dag[j][node] = false;
            }
        }
        S = (0..dag.len())
            .filter(|&j| !order.contains(&j) && !(dag[j].contains(&true)))
            .collect()
    }
    order
}

/// [priority] takes the cost of each node of a graph and outputs the priority of each node.
pub fn priority(cost: &Vec<u32>, graph: &Vec<Vec<bool>>) -> Vec<u32> {
    let s = graph.len();
    let mut current_priority: Vec<u32> = (0..s)
        .map(|j| {
            let mut has_successor = false;
            for i in 0..s {
                if graph[i][j] {
                    has_successor = true;
                    break;
                }
            }
            if has_successor { 0 } else { cost[j] }
        })
        .collect();
    let mut visited = vec![false; s];
    for j in 0..s {
        if visited[j] {
            current_priority[j] = cost[j];
        }
    }
    let mut scc = 0;
    let mut d;
    while visited.contains(&false) {
        d = 0;
        for j in 0..s {
            if !visited[j] && current_priority[j] > d {
                scc = j;
                d = current_priority[j];
            }
        }
        for j in 0..s {
            if graph[scc][j] && cost[j] + d > current_priority[j] {
                current_priority[j] = cost[j] + d;
            }
        }
        visited[scc] = true;
    }
    current_priority
}