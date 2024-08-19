import pkg_resources
import importlib

# List of packages you want to check
packages = [
    "os", "time", "logging", "flask", "threading", "requests", "numpy", 
    "pandas", "keras", "scikit-learn", "json", "sys", "tensorflow", "PIL"
]

# Create or overwrite the requirements.txt file
with open('requirements.txt', 'w') as f:
    for package in packages:
        if package in ["os", "time", "logging", "threading", "sys", "json"]:
            # These are built-in modules, so we manually specify that they are part of Python's standard library
            f.write(f"{package} (built-in)\n")
            print(f"{package} is a built-in module")
        else:
            try:
                version = pkg_resources.get_distribution(package).version
                f.write(f"{package}=={version}\n")
                print(f"{package} version: {version}")
            except pkg_resources.DistributionNotFound:
                f.write(f"{package} not found\n")
                print(f"{package} not found")
            except Exception as e:
                f.write(f"Error retrieving {package}: {e}\n")
                print(f"Error retrieving {package}: {e}")
