// Author: Jekaterina Jaroslavceva - jejarosl@cern.ch
// Date: January 2024

/*
This TICL plugin implements a graph neural network(and MLP)-based trackster linking algorithm designed and trained by Jekaterina Jaroslavceva. The algorithm is tailored for hadronic interactions in HGCAL.

Inputs:
- CLUE3D tracksters.

Outputs:
- Superclusters represented as vectors of trackster IDs.

Algorithm Description:
The plugin implements an inference of the graph neural network (GNN) (or potentially, an MLP algorithm) for superclustering tracksters in the HGCAL detector. The process involves the following steps:

1. Construction of a TICLGraph object representing the spatial relationships between tracksters.
2. Feature extraction from tracksters including their barycenters, eigenvalues, sigmas, energies, and time information.
3. Generation of edge features such as energy differences, spatial and temporal compatibilities between tracksters.
4. Prediction of trackster connections using the GNN model trained for this purpose.
5. Identification of connected components using depth-first search (DFS) on the predicted graph.
6. Assembly of superclusters by merging tracksters within each connected component.

Parameters:
- nnVersion: A tag indicating the version of the DNN model used for trackster linking.
- nnWorkingPoint: The threshold score for considering a trackster pair as connected.
- deltaEtaWindow: The size of the delta eta window used to filter candidate pairs for superclustering.
- deltaPhiWindow: The size of the delta phi window used to filter candidate pairs for superclustering.
*/

#include <unordered_map>
#include <vector>
#include <string>

#include "RecoHGCal/TICL/interface/TICLGraph.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingbyGNN.h"
#include "RecoHGCal/TICL/interface/GNNUtils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace ticl;
// Define constants for feature shapes
const int FEATURE_SHAPE_GNN_V1 = 28;
const int FEATURE_SHAPE_MLP = 30;
const int NUM_EDGE_FEATURES = 7;

// Define the enum for DNN versions
enum class DNNVersion {
    GNN_V1,
    MLP_EDGE_FEATURES,
    MLP_NO_EDGE_FEATURES
};

// Define constants for DNN versions
std::unordered_map<std::string, DNNVersion> DNN_VERSION_MAP = {
    {"gnn_v1", DNNVersion::GNN_V1},
    {"mlp_edge_features", DNNVersion::MLP_EDGE_FEATURES},
    {"mlp_no_edge_features", DNNVersion::MLP_NO_EDGE_FEATURES}
};

// Define a map to store DNN input configurations
std::unordered_map<DNNVersion, std::pair<std::vector<std::string>, int>> dnnInputConfigurations = {
    {DNNVersion::GNN_V1, {{"features", "edge_index", "edge_features"}, FEATURE_SHAPE_GNN_V1}},
    {DNNVersion::MLP_EDGE_FEATURES, {{"features", "edge_index", "edge_features"}, FEATURE_SHAPE_MLP}},
    {DNNVersion::MLP_NO_EDGE_FEATURES, {{"features", "edge_index"}, FEATURE_SHAPE_MLP}}
};

void TracksterLinkingbyGNN::initialize(const HGCalDDDConstants *hgcons,
                                             const hgcal::RecHitTools rhtools,
                                             const edm::ESHandle<MagneticField> bfieldH,
                                             const edm::ESHandle<Propagator> propH) {};

TracksterLinkingbyGNN::TracksterLinkingbyGNN(const edm::ParameterSet &ps, edm::ConsumesCollector iC, cms::Ort::ONNXRuntime const *onnxRuntime)
    : TracksterLinkingAlgoBase(ps, iC, onnxRuntime),
      nnVersion_(ps.getParameter<std::string>("nnVersion")),
      nnWorkingPoint_(ps.getParameter<double>("nnWorkingPoint")),
      deltaEtaWindow_(ps.getParameter<double>("deltaEtaWindow")),
      deltaPhiWindow_(ps.getParameter<double>("deltaPhiWindow"))
{
    assert(onnxRuntime_ && "TracksterLinkingbySuperClustering : ONNXRuntime was not provided, the model should have been set in onnxModelPath in the plugin config");
}

/**
 * resultTracksters : all tracksters (including those not in Superclusters (SC)).
 * outputSuperclusters : indices into resultsTracksters. Include tracksters not in SC as one-element vectors.
 * linkedTracksterIdToInputTracksterId : map indices from output to input.
 */
void TracksterLinkingbyGNN::linkTracksters(const Inputs &input, std::vector<Trackster> &resultTracksters,
                                           std::vector<std::vector<unsigned int>> &outputSuperclusters,
                                           std::vector<std::vector<unsigned int>> &linkedTracksterIdToInputTracksterId)
{
    LogDebug("HGCalTICLSuperclustering") << "Linking Algo by " << nnVersion_ << std::endl;
    // Create trackster graph
    auto TICL_graph_result = produce_ticl_graph(input.tracksters, deltaEtaWindow_, deltaPhiWindow_);

    // Access the TICLGraph object through the unique pointer
    TICLGraph *ticlGraph = TICL_graph_result.get();

    // Access the corresponding DNNVersion enum
    DNNVersion versionEnum = DNN_VERSION_MAP[nnVersion_];

    // Access the input configuration based on the DNN version
    std::vector<std::string> input_names;
    int shapeFeatures;

    auto dnnConfig = dnnInputConfigurations.find(versionEnum);
    if (dnnConfig != dnnInputConfigurations.end()) {
        // Valid DNN version found, retrieve input names and shape features
        input_names = dnnConfig->second.first;
        shapeFeatures = dnnConfig->second.second;
    } else {
        // Invalid DNN version
        edm::LogError("HGCalTICL_TracksterLinkingByGNN") << "Architecture not defined: " << nnVersion_ ;
        return;
    }
    
    // Print the retrieved input configuration
    std::cout << "Input names:";
    for (const auto& name : input_names) {
        std::cout << " " << name;
    }
    std::cout << std::endl;
    std::cout << "Shape of features: " << shapeFeatures << std::endl;
    
    // Array of data to be filled as a network input. Should be a float array of flattened values.
    cms::Ort::FloatArrays data;
    // Network input shapes.
    std::vector<std::vector<int64_t>> input_shapes;
    

    auto const &tracksters = input.tracksters;
    auto const &layerClusters = input.layerClusters;
    long int N = input.tracksters.size();

    LogDebug("HGCalTICL_TracksterLinkingByGNN") << "Number of tracksters in event: " << N << std::endl;

    if (N < 2) {
        // do not run the network - return the original tracksters
        LogDebug("HGCalTICL_TracksterLinkingByGNN") << "Number of tracksters less than 2 - no linking is done." << std::endl;
        
        linkedTracksterIdToInputTracksterId.resize(N);
        std::vector<unsigned int> linkedTracksters;

        for (int trackster_id = 0; trackster_id < N; trackster_id++) {
            linkedTracksterIdToInputTracksterId[trackster_id].push_back(trackster_id);
            linkedTracksters.push_back(resultTracksters.size());
            resultTracksters.push_back(input.tracksters[trackster_id]);
            outputSuperclusters.push_back(linkedTracksters);
        }
        return;
    }

    std::vector<float> features;
    std::vector<float> edge_features;
    
    std::vector<float> node_degrees;
    std::vector<float> degree_centr;

    for (unsigned i = 0; i < N; ++i) {

        const auto ts = tracksters[i];
        const Vector &barycenter = ts.barycenter();
        const Vector eigenvector0 = ts.eigenvectors(0);
        const std::array<float, 3> eigenvalues = ts.eigenvalues();
        const std::array<float, 3> sigmasPCA = ts.sigmasPCA();

        // Get representative points of the trackster
        const std::vector<unsigned int> &vertices_indices = ts.vertices();

        const auto max_z_lc_it = std::max_element(
            vertices_indices.begin(),
            vertices_indices.end(),
            [&layerClusters](const int &a, const int &b)
            {
                return layerClusters[a].z() > layerClusters[b].z();
            });

        const auto min_z_lc_it = std::min_element(
            vertices_indices.begin(),
            vertices_indices.end(),
            [&layerClusters](const int &a, const int &b)
            {
                return layerClusters[a].z() > layerClusters[b].z();
            });

        int num_hits = 0;
        std::set<float> lc_z_unique;
        
        for (auto &lc_idx : ts.vertices()) {
            auto lc = layerClusters[lc_idx];
            num_hits += lc.size();
            lc_z_unique.insert(lc.z());
        }
        auto length = lc_z_unique.size();

        const reco::CaloCluster &min_z_lc = layerClusters[*min_z_lc_it];
        const reco::CaloCluster &max_z_lc = layerClusters[*max_z_lc_it];

        // Features
        features.push_back(barycenter.x());         // 0:  barycenter x
        features.push_back(barycenter.y());         // 1:  barycenter y
        features.push_back(barycenter.z());         // 2:  barycenter z

        features.push_back(barycenter.eta());       // 3: trackster_barycenter_eta
        features.push_back(barycenter.phi());       // 4: trackster_barycenter_phi

        features.push_back(eigenvector0.x());       // 5: eVector0_x
        features.push_back(eigenvector0.y());       // 6: eVector0_y
        features.push_back(eigenvector0.z());       // 7: eVector0_z

        features.push_back(eigenvalues[0]);         // 8: EV1
        features.push_back(eigenvalues[1]);         // 9: EV2
        features.push_back(eigenvalues[2]);         // 10: EV3

        features.push_back(sigmasPCA[0]);           // 11: sigmaPCA1
        features.push_back(sigmasPCA[1]);           // 12: sigmaPCA2
        features.push_back(sigmasPCA[2]);           // 13: sigmaPCA3

        features.push_back(ts.vertices().size());   // 14: number of LCs per trackster
        features.push_back(num_hits);               // 15: number of hits per trackster

        features.push_back(ts.raw_energy());        // 16: raw_energy
        features.push_back(ts.raw_em_energy());     // 17: raw_em_energy

        features.push_back(ts.id_probabilities(0)); // 18: photon probability
        features.push_back(ts.id_probabilities(1)); // 19: electron probability
        features.push_back(ts.id_probabilities(2)); // 20: muon probability
        features.push_back(ts.id_probabilities(3)); // 21: neutral_pion probability
        features.push_back(ts.id_probabilities(4)); // 22: charged_hadron probability
        features.push_back(ts.id_probabilities(5)); // 23: neutral_hadron probability

        features.push_back(min_z_lc.z());           // 24: z_min (minimum z of constituent LCs)
        features.push_back(max_z_lc.z());           // 25: z_max (maximum z of constituent LCs)
        features.push_back(length);                 // 26: trackster length from minimum to maximum element in terms of layers
        // 27 - will be added later in code (if not a gnn) - node degrees
        // 28 - will be added later in code (if not a gnn) - degrees centrality
        features.push_back(ts.time());             // 27(29): time
    }

    std::vector<float> edges_src;
    std::vector<float> edges_dst;
    int node_degree_max = 0;
    
    for (int i = 0; i < N; i++){
        const auto &ts_i = tracksters[i];

        std::vector<float> vertex_energy_i;
        std::vector<float> vertex_i_x;
        std::vector<float> vertex_i_y;
        std::vector<float> vertex_i_z;

        const Vector &eigenvector0_i = ts_i.eigenvectors(0);

        for (auto &lc_idx : ts_i.vertices()){
            auto lc = layerClusters[lc_idx];
            vertex_energy_i.push_back(lc.energy());
            vertex_i_x.push_back(lc.x());
            vertex_i_y.push_back(lc.y());
            vertex_i_z.push_back(lc.z());
        }
        int node_degree = ticlGraph->getNode(i).getInner().size();
        node_degrees.push_back(node_degree);
        
        if (node_degree > node_degree_max){
          node_degree_max = node_degree;
        }

        for (auto &i_neighbour : ticlGraph->getNode(i).getInner()){

            const auto &ts_j = tracksters[i_neighbour];

            std::vector<float> vertex_energy_j;
            std::vector<float> vertex_j_x;
            std::vector<float> vertex_j_y;
            std::vector<float> vertex_j_z;

            const Vector &eigenvector0_j = ts_j.eigenvectors(0);

            for (auto &lc_idx : ts_j.vertices()){
                auto lc = layerClusters[lc_idx];
                vertex_energy_j.push_back(lc.energy());
                vertex_j_x.push_back(lc.x());
                vertex_j_y.push_back(lc.y());
                vertex_j_z.push_back(lc.z());
            }

            // Create an edge between the tracksters
            edges_src.push_back(static_cast<float>(i_neighbour));
            edges_dst.push_back(static_cast<float>(i));

            auto delta_en = std::abs(ts_i.raw_energy() - ts_j.raw_energy());
            auto delta_z = std::abs(ts_i.barycenter().z() - ts_j.barycenter().z());

            // Find energy thresholds
            float e_thr_i = 0.01 * *std::max_element(vertex_energy_i.begin(), vertex_energy_i.end());
            float e_thr_j = 0.01 * *std::max_element(vertex_energy_j.begin(), vertex_energy_j.end());

            // Filter vertices based on energy threshold
            std::vector<std::vector<float>> lcs_i;
            for (int k = 0; k < (int)vertex_energy_i.size(); k++){
                float x = vertex_i_x[k];
                float y = vertex_i_y[k];
                float z = vertex_i_z[k];
                float e = vertex_energy_i[k];
                if (e >= e_thr_i)
                    lcs_i.push_back({x, y, z});
            }

            std::vector<std::vector<float>> lcs_j;
            for (int k = 0; k < (int)vertex_energy_j.size(); k++){
                float x = vertex_j_x[k];
                float y = vertex_j_y[k];
                float z = vertex_j_z[k];
                float e = vertex_energy_j[k];
                if (e >= e_thr_j)
                    lcs_j.push_back({x, y, z});
            }

            // Calculate distances
            auto dists = calculate_dist(lcs_i, lcs_j);
            float min_dist = *std::min_element(dists.begin(), dists.end());
            float max_dist = *std::max_element(dists.begin(), dists.end());

            // Calculate delta_r
            float delta_r = std::sqrt(std::pow((ts_i.barycenter().x() - ts_j.barycenter().x()), 2) +
                                      std::pow((ts_i.barycenter().y() - ts_j.barycenter().y()), 2));

            // Calculate spatial compatibility
            float prod = eigenvector0_i.x() * eigenvector0_j.x() +
                         eigenvector0_i.y() * eigenvector0_j.y() +
                         eigenvector0_i.z() * eigenvector0_j.z();
            prod = (prod > 1) ? 1 : prod;
            prod = (prod < 0) ? 0 : prod;
            float spatial_comp = std::acos(prod);

            // Calculate time compatibility
            float time_comp = 0.0;
            if (ts_i.time() > -99 && ts_j.time() > -99)
                time_comp = std::abs(ts_i.time() - ts_j.time());
            else
                time_comp = -10.0;

            edge_features.push_back(delta_en);      // 0 - energy_difference
            edge_features.push_back(delta_z);       // 1 - delta_z
            edge_features.push_back(min_dist);      // 2 - min_dist
            edge_features.push_back(max_dist);      // 3 - max_dist
            edge_features.push_back(delta_r);       // 4 - delta_r
            edge_features.push_back(spatial_comp);  // 5 - spatial_comp
            edge_features.push_back(time_comp);     // 6 - time_comp
        }
    }

    auto numEdges = static_cast<int>(edges_src.size());

    if (numEdges < 1) {
        LogDebug("HGCalTICL_TracksterLinkingByGNN") << "No edges for the event - no linking is done." << std::endl;
        // Do not run the network - return the original tracksters
        linkedTracksterIdToInputTracksterId.resize(N);
        std::vector<unsigned int> linkedTracksters;

        for (int trackster_id = 0; trackster_id < N; trackster_id++)
        {
            linkedTracksterIdToInputTracksterId[trackster_id].push_back(trackster_id);
            linkedTracksters.push_back(resultTracksters.size());
            resultTracksters.push_back(input.tracksters[trackster_id]);
            outputSuperclusters.push_back(linkedTracksters);
        }
        return;
    }
    
    // If we use MLP - add additional two featrues
    if (nnVersion_ == "mlp_edge_features" || nnVersion_ == "mlp_no_edge_features"){
      auto degrees_centr = countDegree(edges_src, edges_dst, N);
      for (int i = 0; i < N; i++)
      {    
         features.insert(features.begin() + 27 * (i+1), node_degrees[i]/node_degree_max); // 27 - inner node degrees
         features.insert(features.begin() + 28 * (i+1), degrees_centr[i]); // 28 - degree centrality
      }
    }
    
    input_shapes.push_back({1, N, shapeFeatures});
    data.emplace_back(features);

    input_shapes.push_back({1, 2, numEdges});

    data.emplace_back(edges_src);
    for (auto &dst : edges_dst)
    {
        data.back().push_back(dst);
    }

    if (nnVersion_ != "mlp_no_edge_features"){
      input_shapes.push_back({1, numEdges, NUM_EDGE_FEATURES});
      data.emplace_back(edge_features);
    }
    
    std::vector<float> edge_predictions = onnxRuntime_->run(input_names, data, input_shapes)[0];
    
    LogDebug("HGCalTICL_TracksterLinkingByGNN") << "Network output shape is " << edge_predictions.size() << std::endl;
    // Update the TICL graph weights
    for (int i = 0; i < static_cast<int>(edge_predictions.size()); i++){
            LogDebug("HGCalTICL_TracksterLinkingByGNN") << "Network output for edge " << data[1][i] << "-" << data[1][numEdges + i] << " is: " << edge_predictions[i] << std::endl;
            ticlGraph->setEdgeWeight(data[1][i], data[1][numEdges + i], edge_predictions[i]);
    }
    
    auto connected_components = ticlGraph->findSubComponents(nnWorkingPoint_);
    
    
    int id = 0;
    linkedTracksterIdToInputTracksterId.resize(connected_components.size());
    
    for (auto &component : connected_components)
    {
        std::vector<unsigned int> linkedTracksters;
        Trackster outTrackster;
      
        for (auto &trackster_id : component)
        {
            LogDebug("HGCalTICL_TracksterLinkingByGNN") << "Component " << id << ": trackster id " << trackster_id << std::endl;
            linkedTracksterIdToInputTracksterId[id].push_back(trackster_id);
            outTrackster.mergeTracksters(input.tracksters[trackster_id]);
        }
        id++;
        linkedTracksters.push_back(resultTracksters.size());
        resultTracksters.push_back(outTrackster);
        // Store the linked tracksters
        outputSuperclusters.push_back(linkedTracksters);
    }
}

void TracksterLinkingbyGNN::fillPSetDescription(edm::ParameterSetDescription &desc)
{
    TracksterLinkingAlgoBase::fillPSetDescription(desc); // adds algo_verbosity
    desc.add<edm::FileInPath>("onnxModelPath", edm::FileInPath("RecoHGCal/TICL/data/tf_models/gnn_linking_model.onnx"))
        ->setComment("Path to GNN (as ONNX model)."); // gnn_linking_model.onnx, mlp_linking_model.onnx or mlp_linking_model_with_edge_features.onnx
    desc.add<std::string>("nnVersion", "gnn_v1") // gnn_v1, mlp_no_edge_features, mlp_edge_features
        ->setComment("DNN version tag.");
    desc.add<double>("nnWorkingPoint", 0.95)
        ->setComment("Working point of the DNN (in [0, 1]). DNN score above WP will attempt to supercluster.");
    desc.add<double>("deltaEtaWindow", 0.1)
        ->setComment("Size of delta eta window to consider for superclustering. Candidate pairs outside this window are not considered for DNN inference.");
    desc.add<double>("deltaPhiWindow", 0.5)
        ->setComment("Size of delta phi window to consider for superclustering. Candidate pairs outside this window are not considered for DNN inference.");
}