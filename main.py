from sdv.datasets.demo import get_available_demos
from sdv.datasets.demo import download_demo
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import get_column_pair_plot
import os
import pandas as pd
import json
# os.environ['CURL_CA_BUNDLE']=os.path.join(os.getcwd(), "root.pem")
# os.environ['SSL_CERT_FILE']=os.path.join(os.getcwd(), "root.pem")
# os.environ['REQUESTS_CA_BUNDLE']=os.path.join(os.getcwd(), "root.pem")
# os.environ['NODE_EXTRA_CA_CERTS']=os.path.join(os.getcwd(), "root.pem")


class DataSynthesizer:
    def __init__(self, config_path="config.json"):

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.input_path = self.config.get("input_path", "datasets/")
        self.model_name = self.config.get("model_name", "TabularGAN")
        self.output_path = self.config.get("output_path", "output/")
        self.num_samples = self.config.get("num_samples", 100)

        self.datasets = {}
        self.metadata = {}
        self.synthesizers = {}
        self.synthetic_data = {}

    def load_data(self):
        self.datasets = load_csvs(
            folder_name=self.input_path,
            read_csv_parameters={
                'skipinitialspace': True,
                'encoding': 'utf_32'
            }
        )

        self.metadata = {}
        for dataset_name in self.datasets.keys():
            self.metadata[dataset_name] = Metadata.detect_from_dataframe(
                    data=self.datasets[dataset_name],
                    table_name=dataset_name
            )
        
        return list(self.datasets.keys())
    
    def create_synthesizer(self, dataset_name):
        """Creating a synthesizer"""
        if self.model_name == "GaussianCopula":
            self.synthesizer = GaussianCopulaSynthesizer(
                self.metadata[dataset_name],
                enforce_min_max_values=True,
                enforce_rounding=False,
                default_distribution='gaussian_kde'
                )
        elif self.model_name == "TabularGAN":
            self.synthesizer = CTGANSynthesizer(
                self.metadata[dataset_name],
                enforce_rounding=False,
                epochs=100,
                verbose=True
                )
        elif self.model_name == "TVAE":
            self.synthesizer = TVAESynthesizer(
                self.metadata[dataset_name],
                enforce_min_max_values=True,
                enforce_rounding=False,
                epochs=100
                )
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        self.synthesizer.fit(self.datasets[dataset_name])
        print(f"Synthesizer created for {dataset_name} using {self.model_name} model.")
        self.synthesizers[dataset_name] = self.synthesizer
        
    def generate_samples(self, dataset_name):
        """Generate synthetic samples"""
        print(f"Generating {self.num_samples} synthetic samples...")
        synthetic_data = self.synthesizers[dataset_name].sample(num_rows=self.num_samples)
        self.synthetic_data[dataset_name] = synthetic_data
        print(f"Generated {len(synthetic_data)} synthetic rows")

    def evaluate_data(self, dataset_name):
        """Evaluate the synthetic data"""
        # Implement your evaluation logic here
        self.diagnostic = run_diagnostic(
            real_data=self.datasets[dataset_name],
            synthetic_data=self.synthetic_data[dataset_name],
            metadata=self.metadata[dataset_name]
            )
        print(f"Diagnostic report for {dataset_name}:")
        print(self.diagnostic)

        self.quality_report = evaluate_quality(
            real_data=self.datasets[dataset_name],
            synthetic_data=self.synthetic_data[dataset_name],
            metadata=self.metadata[dataset_name]
            )
        print(f"Quality report for {dataset_name}:")
        print(self.quality_report)
        print(f"Column wise scores for {dataset_name}:")
        print(self.quality_report.get_details('Column Shapes'))

        # Generate individual column plots
        for column in self.datasets[dataset_name].columns:
            try:
                fig = get_column_plot(
                    real_data=self.datasets[dataset_name],
                    synthetic_data=self.synthetic_data[dataset_name],
                    column_name=column,
                    metadata=self.metadata[dataset_name]
                )
                
                plot_path = os.path.join(self.output_path, f"{dataset_name}_{column}_plot.png")
                fig.savefig(plot_path)
                print(f"Plot for column '{column}' saved to {plot_path}")
            except Exception as e:
                print(f"Error generating plot for column '{column}': {str(e)}")

        pass

    def save_synthetic_data(self, dataset_name):
        """Save synthetic data to CSV"""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        
        output_file = os.path.join(
            self.output_path, 
            f"{dataset_name}_synthetic.csv"
        )
        
        self.synthetic_data[dataset_name].to_csv(output_file, index=False)
        print(f"Saved synthetic data to {output_file}")
    
    def run_pipeline(self):
        """Run the full pipeline"""
        # Load data
        dat_name = self.load_data()
        synthesizers = {}
        
        for dataset_name in dat_name:
            
            # Create and train synthesizer
            self.create_synthesizer(dataset_name)
            
            # Generate synthetic samples
            self.generate_samples(dataset_name)

            self.evaluate_data(dataset_name)
            
            # Save results
            self.save_synthetic_data(dataset_name)
        
        pass




if __name__ == "__main__":
    pipeline = DataSynthesizer(config_path="config.json")
    pipeline.run_pipeline()