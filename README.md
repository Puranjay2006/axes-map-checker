# 🗺️ Axes Systems: AI Map Validator

An automated topology checker for road networks that detects undershoots, gaps, and connectivity errors using dangling node detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

- **File Upload**: Upload `.wkt` or `.txt` files containing LINESTRING geometries
- **Demo Data**: Built-in sample data for instant testing
- **AI-Powered Detection**: Rule-based dangling node analysis to find topology errors
- **Interactive Map**: Pan/zoom visualization with Folium
- **Error Classification**: High/Medium severity ratings based on gap distance
- **Export Results**: Download error report as CSV

## 🧠 How It Works

The validator uses a **Dangling Node Detection** algorithm:

1. **Extract Endpoints**: Get start/end points of all road segments
2. **Count Degrees**: Track how many roads share each point
3. **Find Dangles**: Points appearing only once (degree = 1) are potential issues
4. **Measure Gaps**: Calculate distance from each dangle to nearest other road
5. **Classify Errors**:
   - Gap < 2.5 units → 🔴 **UNDERSHOOT** (High confidence error)
   - Gap 2.5-5 units → 🟠 **POTENTIAL_GAP** (Medium confidence)
   - Gap > 5 units → Valid dead-end (ignored)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/axes-map-checker.git
cd axes-map-checker

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🌐 Deploy to Streamlit Cloud (FREE!)

### Step-by-Step:

1. **Create a GitHub Repository**
   - Go to [github.com/new](https://github.com/new)
   - Name it `axes-map-checker`
   - Make it public
   - Upload `app.py` and `requirements.txt`

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository: `axes-map-checker`
   - Main file path: `app.py`
   - Click **"Deploy!"**

3. **Get Your Live URL**
   - Wait ~30 seconds for deployment
   - Your app will be live at: `https://axes-map-checker.streamlit.app`

## 📁 Project Structure

```
axes-map-checker/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Geometry Engine | Shapely |
| Map Visualization | Folium + streamlit-folium |
| Data Processing | Pandas |

## 📊 Input Format

The validator accepts WKT (Well-Known Text) files with LINESTRING geometries:

```
LINESTRING(x1 y1, x2 y2, x3 y3, ...)
LINESTRING(x1 y1, x2 y2, ...)
```

Coordinates can be in any projected coordinate system (meters, feet, etc.)

## 🎯 Use Cases

- **GIS Quality Control**: Validate digitized road networks
- **Data Cleaning**: Find and fix topology errors before analysis
- **Map Production**: Ensure connectivity for routing applications

## 📜 License

MIT License - Feel free to use and modify!

---

**Built for IIT Mandi Hackathon 3.0** | Team Axes Systems
