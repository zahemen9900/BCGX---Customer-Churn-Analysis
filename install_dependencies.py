import subprocess
import sys
def update_pip():
    """
    Updates pip to the latest version.
    """
    try:
        print("Updating pip to the latest version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("Pip updated successfully.")
    except subprocess.CalledProcessError:
        print("Failed to update pip. Please check your pip or internet connection.")
    except Exception as e:
        print(f"An unexpected error occurred while updating pip: {e}")


def install_packages(packages):
    """
    Installs a list of Python packages using pip.

    Args:
        packages (list): A list of package names to install.
    """
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please check your pip or internet connection.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    update_pip()

if __name__ == "__main__":
    # List of packages for your ML workflow
    ml_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "ipython",
        "optuna",
        "tqdm",
        "joblib",
        "imblearn",
        "catboost",
        #"tensorflow",
    #     "torch",
    #    "torchvision",
        #"jupyter",
       # "jupyterlab",
        # "notebook",
        # "requests",
        # "beautifulsoup4",
        # "openpyxl",
        # "xlrd",
        # "statsmodels",
        # "xgboost",
        # "lightgbm",

    ]

    # Installing the packages
    print("Starting the installation process...")
    install_packages(ml_packages)
    print("All packages installed successfully!")
