import argparse

def parse_arguments():
    """   
    Small parser to make main.py better usable from the command line
    """

    parser = argparse.ArgumentParser(description="VQE Geometry Optimization")
    
    parser.add_argument(
        "-m", "--molecule",
        type=str,
        default="H2",
        choices=["H2","LiH","HF","H2O"],
        help="Molecule to calculate"
    )
    return parser.parse_args()

