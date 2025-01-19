# TableDC

This repo provides the following:
The implementation of TableDC, a deep clustering algorithm for data cleaning and integration, specifically schema inference, entity resolution, and domain discovery.

This project comprises two major components, each addressing different data integration challenges: schema inference, entity resolution, and domain discovery:
* For schema inference, we use [SBERT](https://www.sbert.net/docs/hugging_face.html), [FastText](https://fasttext.cc/docs/en/crawl-vectors.html), USE, and [TabTransformer](https://github.com/jrzaurin/pytorch-widedeep) to embed tables.
* For entity resolution, we employ [EmbDi](https://gitlab.eurecom.fr/cappuzzo/embdi), USE and [SBERT](https://www.sbert.net/docs/hugging_face.html) to embed rows.
* For domain discovery, we utilize [EmbDi](https://gitlab.eurecom.fr/cappuzzo/embdi), T5, USE, [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) and [SBERT](https://www.sbert.net/docs/hugging_face.html) to embed columns.
* The generated dense embedding matrix `(X.txt)` will then serve as the input for clustering in the TableDC.

## Prerequisites
* Python 3.x
* PyTorch
* NumPy
* scikit-learn

## Dataset
The dataset comprises embeddings stored in `data/X.txt` and ground truth labels in `data/label.txt`.

- **Special thanks** to the authors (of baseline deep clustering and embedding approaches mentioned in the paper) for providing their implementations publicly available.

## Steps for Reproducing Results

This demo outlines steps to reproduce results for schema inference, entity resolution and domain discovery with TableDC. One ready-to-use vector X.txt (for schema Inference) is given in `data/`. Just use Step 3 below to get the clustering results.

### Schema Inference

1. **Schema-Level Web Tables Data (SBERT + TableDC):**
   - Process `schema inference/schema + instances/Preprocessing.ipynb` to extract schema-level information from tables.

2. **Generating Embedding Matrix:**
   - Use the generated `TextPre1.csv` to produce a dense embedding matrix (`X.txt`) with SBERT by running `schema inference/schema only/SBERT+FastText.py`.

3. **Clustering with TableDC:**
   - Utilize `X.txt` feature vector for clustering in TableDC:
     - Navigate to the `data/` and run the pretraining script `(pretrain_ae.py): python pretrain_ae.py --pretrain_path data/X.txt`.
     - Ensure that the pretrained model `(X.pkl)` and dataset `(X.txt and label.txt)` are in the `data/ directory`.
     - Run the script `(TableDC.py): python TableDC.py --pretrain_path data/X.pkl --name X`.
     - Update `nb_dimension = 768` accordingly (for SBERT use 768).

### Entity Resolution

1. **Row Embedding Matrix for Entity Resolution:**
   - Run `entity resolution/ER.py` to obtain row embedding matrix (`X.txt`) using EmbDi, and `entity resolution/ER_SBERT/ER_SBERT.py` for row embeddings with SBERT.

2. **Clustering for Entity Resolution:**
   - Repeat step (3) from Schema Inference with the row embedding matrix as input.

### Domain Discovery

- For domain discovery, refer to the respective folders (`schema inference/`, `entity resolution/`, and `domain discovery/`) or `full_data/` for a combination of all embeddings (except DD due to size limits). Ready-to-use embeddings are provided for FastText.
- Example: Compile `domain discovery/DD.py` for column embedding matrix (`X.txt`) using EmbDi, and `domain discovery/DD_SBERT(H+B)/DD_SBERT

## Standard Clustering Algorithms

- To get results with standard clustering algorithms, execute `SC/SC.py`.

## Note

1. **Data Preparation:**
   Due to storage limitations, please unzip `Tables.zip` before compiling the schema inference code.

2. **Computational Environment:**
   Due to different levels of precision in floating-point arithmetic and architectural aspects of different GPUs and CPUs, the resulting values can be slightly different. However, the overall results will remain consistent.

## Model Architecture and Training Configuration

### Autoencoder Architecture

The Autoencoder (AE) in TableDC contains four main layers, with each layer's size specified as follows:
- **Encoder Layer:**
  - Input Dimension: `n_input`
  - Output Dimension: `n_enc_1`
- **Latent Layer:**
  - Input Dimension from Encoder Layer: `n_enc_1`
  - Output Dimension (Latent Space): `n_z`
- **Decoder Layer:**
  - Input Dimension from Latent Layer: `n_z`
  - Output Dimension: `n_dec_1`
- **Output Layer:**
  - Input Dimension from Decoder Layer: `n_dec_1`
  - Output Dimension (Reconstructed Input): `n_input`

### Training Configuration

- **Optimization Algorithm:** Adam optimizer.
- **Learning Rate:** Configurable via `--lr`. Default set to `1e-3`.
- **Loss Function:** Combination of Kullback–Leibler divergence and Mean Squared Error loss.
- **Number of Epochs:** Please see Section 4.1.4 in the paper.
   
### Additional domain discovery clustering results (TableDC vs. existing DC methods)

<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">Camera</th>
    <th colspan="4">Monitor</th>
  </tr>
  <tr>
    <th colspan="2">EmbDi</th>
    <th colspan="2">FastText</th>
    <th colspan="2">EmbDi</th>
    <th colspan="2">FastText</th>
  </tr>
  <tr>
    <th></th>
    <th>ARI</th>
    <th>ACC</th>
    <th>ARI</th>
    <th>ACC</th>
    <th>ARI</th>
    <th>ACC</th>
    <th>ARI</th>
    <th>ACC</th>
  </tr>
  <tr>
    <td>K-means</td>
    <td>0.13</td>	
    <td>0.16	</td>
   <td>0.48	</td>
   <td>0.49	</td>
   <td>0.06</td>
   <td>0.13</td>
  <td> 0.41</td>
  <td> 0.46</td>
  </tr>
  <tr>
    <td>DBSCAN</td>
    <td>0.016</td>	<td>0.13</td>	 <td>0.01</td>	<td>0.17</td>	<td>0.002</td>	<td>0.07</td>	<td>0.01</td>	<td>0.15</td>
  </tr>
  <tr>
    <td>Birch</td>
    <td>0.03</td>	<td>0.14</td>	<td>0.22</td>	<td>0.43</td>	<td>0.04</td>	<td>0.13</td>	<td>0.25</td>	<td>0.40</td>
  </tr>
  <tr>
    <td>SHGP</td>
   <td>0.09</td>	<td>0.14</td>	<td>0.45</td>	<td>0.48</td>	<td>0.05</td>	<td>0.11</td>	<td>0.40</td>	<td>0.46</td>
  </tr>
  <tr>
    <td>DFCN</td>
    <td>0.12</td>	<td>0.15</td>	<td>0.47</td>	<td>0.47</td>	<td>N/A</td>	<td>N/A</td>	<td>N/A</td>	<td>N/A</td>
  </tr>
  <tr>
    <td>EDESC</td>
   	 <td>0.09</td>	 <td>0.16</td>	 <td>0.38</td>	 <td>0.46</td>	 <td>0.06</td>	 <td>0.12</td>	 <td>0.35</td>	 <td>0.35</td>
  </tr>
  <tr>
    <td>SDCN</td>
   	 <td>0.05</td>	 <td>0.17</td>	 <td>0.56</td>	 <td>0.54</td>	 <td>0.05</td>	 <td>0.13</td>	 <td>0.42</td>	 <td>0.46</td>
  </tr>
  <tr>
    <td>TableDC</td>
    	<td>**0.14**</td>	<td>**0.19**</td>	<td>**0.59**</td>	<td>**0.56**</td>	<td>0.06</td>	<td>0.13</td>	<td>**0.47**</td>	<td>**0.47**</td>
  </tr>
</table>
