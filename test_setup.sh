#!/bin/bash
# Quick test script for the Fast Neural Style Transfer project
# Run this after cloning to verify the setup

echo "=========================================="
echo "Fast Neural Style Transfer - Quick Test"
echo "CSE 311 AI Project"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python installation..."
python3 --version || { echo "❌ Python not found!"; exit 1; }
echo "✓ Python found"
echo ""

# Check if virtual environment is recommended
echo "2. Checking environment setup..."
if [ -d "venv" ] || [ -d ".venv" ]; then
    echo "✓ Virtual environment detected"
else
    echo "ℹ️  No virtual environment detected (recommended but optional)"
    echo "   Create one with: python3 -m venv venv && source venv/bin/activate"
fi
echo ""

# Test imports
echo "3. Testing Python module imports..."
python3 -c "
try:
    import sys
    sys.path.insert(0, 'src')
    from transformer import TransformerNet
    from vgg_loss import StyleTransferLoss
    from datasets import CocoDataset
    print('✓ All modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('   Install dependencies: pip install -r requirements-min.txt')
    sys.exit(1)
" || exit 1
echo ""

# Check for sample images
echo "4. Checking for sample images..."
if [ -f "picasso_selfportrait.jpg" ]; then
    echo "✓ Style image found: picasso_selfportrait.jpg"
else
    echo "⚠️  Style image not found: picasso_selfportrait.jpg"
fi

if [ -f "japanese_garden.jpg" ]; then
    echo "✓ Content image found: japanese_garden.jpg"
else
    echo "⚠️  Content image not found: japanese_garden.jpg"
fi
echo ""

# Check directory structure
echo "5. Verifying directory structure..."
for dir in src notebooks models archive; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "❌ Missing: $dir/"
    fi
done
echo ""

# Summary
echo "=========================================="
echo "Setup Verification Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install dependencies (if not already done):"
echo "   pip install -r requirements-min.txt"
echo ""
echo "2. Download COCO dataset:"
echo "   wget http://images.cocodataset.org/zips/val2017.zip"
echo "   unzip val2017.zip"
echo ""
echo "3. Train a model:"
echo "   python src/train.py --dataset-path val2017 --style-image picasso_selfportrait.jpg"
echo ""
echo "4. Or use the Colab notebook:"
echo "   Open notebooks/Colab_Train_And_Run.ipynb in Google Colab"
echo ""
echo "=========================================="
