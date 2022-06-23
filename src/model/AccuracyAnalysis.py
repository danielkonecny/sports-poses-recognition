"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for analysis of accuracy measurements across different experiments.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 22. 06. 2022
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'file_path',
        type=str,
        help="Location of CSV file with accuracy data."
    )
    return parser.parse_args()


def print_plot(accuracies):
    print(f"""\n\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={{Comparison of self-supervised and supervised models' accuracy}},
    xlabel={{Portion of dataset used for training}},
    ylabel={{Accuracy on validation dataset}},
    xmin=0, xmax=1,
    ymin=0, ymax=1,
    xtick={{0,0.1,0.2,0.4,0.6,0.8,1}},
    ytick={{0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}},
    legend pos=south east,
    xmajorgrids=true,
    ymajorgrids=true,
    grid style=dashed,
]
\\addplot[color=red,mark=o]
    coordinates {{{accuracies['Self-Supervised']}}};
\\addplot[color=blue,mark=square]
    coordinates {{{accuracies['Supervised']}}};
\\legend{{Self-Supervised, Supervised}}
\\end{{axis}}
\\end{{tikzpicture}}""")


def main():
    args = parse_arguments()
    file_path = Path(args.file_path)

    df = pd.read_csv(file_path)

    accuracies = {}
    models = df['Model'].unique()
    for model in sorted(models):
        plot_accuracy = ""
        portions = df.loc[df['Model'] == model, 'Training Data Portion'].unique()
        for portion in sorted(portions):
            accuracy = df.loc[(df['Training Data Portion'] == portion) & (df['Model'] == model), 'Validation Accuracy']
            mean = accuracy.mean()
            plot_accuracy += f"({portion:.3f},{mean:.4f})"
            print(f"- Model: {model} - Portion: {portion:.3f} - Mean: {mean:.4f}")

        accuracies[model] = plot_accuracy

    print_plot(accuracies)


if __name__ == "__main__":
    main()
