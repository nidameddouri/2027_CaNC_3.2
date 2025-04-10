# 2025 CaNC
CaNC is a classification module for Weka based on the Classifier Nominal Concepts, based on Formal Concept Analysis.

## Installation

### Windows users

1. **Install [Weka](https://waikato.github.io/weka-wiki/downloading_weka/)**

2. **Install [JDK 17 for Windows](https://www.oracle.com/java/technologies/downloads/#java17)**
   > Important: Earlier or later versions may cause compatibility issues

3. **Install [Git for Windows](https://git-scm.com/download/win)** (for source code management)

4. **Install [Apache Ant](https://ant.apache.org/manual/install.html)** (for building the package)

5. **Go to** [Usage](#usage) section to use the classifier

### Linux users

1. **Install [Weka](https://waikato.github.io/weka-wiki/downloading_weka/)**

2. **Install JDK 17:**
   ```
   sudo apt install openjdk-17-jdk
   ```

3. **Install Git:**
   ```
   sudo apt install git
   ```

4. **Install Ant:**
   ```
   sudo apt install ant
   ```

5. Go to [Usage](#usage) section to use the classifier

## Compatibility

CaNC has been tested and is compatible with the following software versions:
- **Operating Systems**:
  - Ubuntu 20.04, 22.04
  - Windows 10, Windows 11
  - macOS Sequoia 15.3.2
- **Weka**: Version 3.8+ (mainly tested with Weka 3.9)
- **JDK**: Version 17
- **Apache Ant**: Version 1.10.2 and above

Please ensure that your system meets these requirements to avoid any issues during installation and operation of the classifier.

## Usage

### Option 1: Using the Distributable Package
1. **Download the CaNC package**:
   - Download the latest copy of this repo on your computer, and navigate to this file :
      ```bash
        package_build/dist/CNC.zip
      ```

1. Clone the repository:
   ```bash
   git clone https://gitlab.com/nidameddouri/2025-canc
   ```


2. **Install Plugin in Weka**:
   - Start Weka, go to `Tools` > `Package Manager` 
   - Click on `File/URL` button and select the downloaded CaNC package
   - Click `Install` and restart Weka

3. **Using the classifier**:
   - In Weka, load your dataset in the "Preprocess" tab
   - Go to the "Classify" tab
   - In the dropdown menu, look for the CNC classifier (it should appear in the rules category)
   - Configure the parameters and run the test

### Option 2: Building from Source

#### Using weka-src
1. Clone the repository:
   ```bash
   git clone https://gitlab.com/nidameddouri/2025-canc
   ```

2. Navigate to the weka-src folder:
   ```bash
   cd weka-src
   ```

3. Compile the project with JDK 17 (VS Code recommended)

4. Launch Weka with your classifier:
   ```bash
   java -jar dist/weka.jar
   ```

#### Using package_build
1. Navigate to the package_build folder:
   ```bash
   cd package_build
   ```

2. Use Ant to build the package:
   ```bash
   ant clean package
   ```

3. The resulting JAR file can be found in package_build/dist and can be added to an existing Weka installation via the package manager as described in Option 1.

Note : Please refer to the README in the package_build folder for detailed instructions on this method.

## Future Enhancements and Limitations

### Future Enhancements
1. **Support for Gallois Connection**:
   - Current implementation isn't working as expected, we need to dive deeper in order to make it work on multiple types

## Authors
> Version 2.0 (10/04/2025):
- Francoise Lapetite (francoise.lapetite@epita.fr)
- Younes Kara Zaitri (younes.kara-zaitri@epita.fr)

Supervised by **Nidà MEDDOURI** (nida.meddouri@epita.fr) and **Fabrice BOISSIER** (fabrice.boissier@epita.fr)