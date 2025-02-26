# AlphaFold环境搭建

## 环境

* Ubuntu 20.04以上
* cmake
* make
* conda python3.9


## hmmer

```bash
wget http://eddylab.org/software/hmmer/hmmer.tar.gz
tar zxf hmmer.tar.gz
cd hmmer-3.3.2
./configure --prefix=<PREFIX>/hmmer
make
make install
source PATH=$PATH:<PREFIX>/hmmer
```

## HHsuite


```bash
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=<PREFIX>/hh-suite ..
cmake --build . && cmake --install .
```

## Kalign

```bash
wget https://github.com/TimoLassmann/kalign/archive/refs/tags/v3.3.5.tar.gz
tar xvf v3.3.5.tar.gz
cd kalign-3.3.5
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=<PREFIX>/kalign ..
cmake --build . && cmake --install .
```

## OpenMM

```bash
conda install -c conda-forge openmm

# 验证OpenMM是否安装成功
python -m openmm.testInstallation
```

## PDBfixer

```bash
pip install git+http://github.com/openmm/pdbfixer.git
```


## AlphaFold

```bash
# 激活所有依赖的环境
export PATH=$PATH:<PREFIX>/hmmer/bin:<PREFIX>/hh-suite/bin:<PREFIX>/kalign/bin
# 克隆AlphaFold库
git clone https://github.com/deepmind/alphafold
cd alphafold

pip install -r requirements.txt
# stereo_chemical_props.txt下载
wget -P alphafold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# 数据库安装
## 下载全部数据库
./scripts/download_all_data.sh <DATA_DIR>

## 如果你的硬盘空间不足，也可以尝试只下载reduced database
./scripts/download_all_data.sh <DATA_DIR> reduced_dbs

mkdir ./output

# 启动推理
python ./run_alphafold.py \
  --fasta_paths=<DATA_DIR>/monomer.fasta \
  --max_template_date=2020-05-14 \
  --db_preset=reduced_dbs \
  --data_dir=<DATA_DIR> \
  --output_dir=./output \
  --uniref90_database_path=<DATA_DIR>/uniref90.fasta \
  --mgnify_database_path=/mnt/e/alphafold_data/mgnify/mgy_clusters_2022_05.fa \
  --template_mmcif_dir=<DATA_DIR>/pdb_mmcif/mmcif_files \
  --obsolete_pdbs_path=<DATA_DIR>/pdb_mmcif/obsolete.dat \
  --use_gpu_relax=1 \
  --small_bfd_database_path=<DATA_DIR>small_bfd/bfd-first_non_consensus_sequences.fasta   \
  --use_precomputed_msas=1 \
  --pdb70_database_path=<DATA_DIR>/pdb70/pdb70
```

