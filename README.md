# Self-Portrait to Pixel Art Converter

This project converts self-portrait images into pixel art using machine learning. It utilizes a pre-trained CycleGAN model developed by Irina Nikolaeva and her colleagues, and builds upon the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) framework.

## Overview

The project takes self-portrait images and transforms them into pixel art style using a machine learning model. The input images were scraped from WikiArt using the [WikiArt Self-Portrait Image Scraper](https://github.com/guyanik/self-portrait-data) repository that I created as well.

## Example

Input Image:
![Input Image](images/Double%20Self-Portrait%2C%201944%20-%20Antonio%20Bueno.png)

Output Image:
![Output Image](pixel-art/pix_Double%20Self-Portrait%2C%201944%20-%20Antonio%20Bueno.png)

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- numpy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/guyanik/selfie-to-pixel-art
cd self-portrait-to-pixel-art
```

2. Install the required CycleGAN framework:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt
cd ..
```

3. Download the pre-trained model:
```bash
curl -L -o latest_net_G_A.pth "https://www.dropbox.com/s/bqc486kajuvetmf/latest_net_G_A.pth?dl=1"
```

## Usage

1. Place your input images in the `images` directory
2. Run the conversion script:
```bash
python image_to_pixel_art.py
```
3. Find the converted images in the `pixel-art` directory with the prefix "pix_"

## Credits

- Original CycleGAN implementation: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Pre-trained model and methodology: [Irina Nikolaeva's Medium article](https://inikolaeva.medium.com/make-pixel-art-in-seconds-with-machine-learning-e1b1974ba572)
- Image scraping tools: [WikiArt Self-Portrait Image Scraper](https://github.com/guyanik/self-portrait-data)

## Acknowledgments

- [Irina Nikolaeva](https://inikolaeva.medium.com/) for the pre-trained model and detailed explanation
- The [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/graphs/contributors) team for the framework
- [WikiArt](https://www.wikiart.org/) for the self-portrait images used in this project

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- H. Görkem Uyanık