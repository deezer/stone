import argparse
import logging

from training_loop import main_loop

logging.basicConfig(level=logging.INFO)


def main(
    n_epochs: int,
    n_steps: int,
    val_steps: int,
    learning_rate: float,
    gin_file: str,
    save_dir: str,
    name: str,
    train_type: str,
    circle_type: int,
) -> None:

    logging.info("PARAMETERS used for TRAINING:")
    logging.info("\t learning_rate: {}".format(learning_rate))
    logging.info("\t epochs: {}".format(n_epochs))
    logging.info("\t training_steps_per_epoch: {}".format(n_steps))
    logging.info("\t validation_steps: {}".format(val_steps))
    logging.info("\t gin_file: {}".format(gin_file))
    logging.info("\t circle_type: {}".format(circle_type))
    logging.info("\t train_type: {}".format(train_type))

    main_loop(
        n_epochs,
        n_steps,
        val_steps,
        learning_rate,
        gin_file,
        save_dir,
        name,
        train_type,
        circle_type,
    )

def input_args() -> None:
    parser = argparse.ArgumentParser(description="ISMIR 24", add_help=True)
    parser.add_argument(
        "-s",
        "--save_dir",
        default=".",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument(
        "-e", "--n-epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "-ts",
        "--train-steps",
        type=int,
        default=512,
        help="steps_per_epoch of each training loop",
    )
    parser.add_argument(
        "-vs",
        "--val-steps",
        type=int,
        default=256,
        help="validation steps for each validation run. MUST be a positive integer",
    )
    parser.add_argument(
            "-lr", 
            "--learning-rate", 
            type=float, 
            default=1e-3, 
            help=""
    )
    parser.add_argument(
        "-g", 
        "--gin-file",
        type=str, 
        default="", 
        help="Path to the config file."
    )
    parser.add_argument(
        "-n",
        "--exp-name",
        type=str,
        default="basic",
        help="name of the exp",
    )
    parser.add_argument(
        "-tt",
        "--train-type",
        type=str,
        default="",
        help="type of training. ks for key signature, ks_mode for key signature and mode",
    )
    parser.add_argument(
        "-c",
        "--circle-type",
        type=int,
        default=7,
        help="circle type for the z transformation, 1 -> circle of semiton, 7-> circle of fifths",
    )
    args = parser.parse_args()

    assert args.train_steps > 0
    assert args.n_epochs > 0
    assert args.val_steps > 0
    assert args.circle_type in [1, 7]
    assert args.train_type in ["ks", "ks_mode"]

    main(
        n_epochs=args.n_epochs,
        n_steps=args.train_steps,
        val_steps=args.val_steps,
        learning_rate=args.learning_rate,
        gin_file=args.gin_file,
        save_dir=args.save_dir,
        name=args.exp_name,
        train_type = args.train_type,
        circle_type=args.circle_type,
    )


if __name__ == "__main__":
    input_args()
