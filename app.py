"""
Axes Systems: AI Map Validator
==============================
A Streamlit web application for detecting topology errors in road networks.
This tool analyzes WKT (Well-Known Text) files containing LineString geometries
and identifies undershoots/gaps using dangling node detection.

Author: Axes Systems Team
Hackathon: IIT Mandi Course Hackathon 3.0
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely import wkt
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
import re
from collections import defaultdict

# =============================================================================
# DEMO DATA - Embedded WKT from Problem 2 - streets_xgen.wkt
# =============================================================================
DEMO_WKT_DATA = """LINESTRING(7071.421606 8585.627528, 7074.672945 8588.813669, 7074.902551 8589.026835, 7075.084535 8589.196346, 7079.265638 8592.884220, 7081.745726 8594.906230, 7087.160750 8598.968277, 7091.114457 8601.621165, 7097.122261 8605.189984, 7104.911244 8609.486740, 7105.304126 8609.703307, 7105.767874 8609.959559, 7144.536643 8631.196101, 7154.035087 8636.400567, 7154.349676 8636.571269, 7154.424397 8636.611861, 7163.899654 8641.749543, 7194.100989 8658.211975, 7204.788850 8664.037795, 7217.453480 8671.075087, 7225.937575 8676.277228, 7229.984882 8679.124346, 7234.367017 8682.488447, 7238.456504 8685.821480, 7243.451150 8690.254299, 7247.565354 8694.303307, 7251.962457 8699.059843, 7255.123087 8702.735811, 7259.170961 8707.866520, 7262.826576 8713.000630, 7262.874709 8713.070759, 7263.406488 8713.854425, 7268.192504 8721.195591, 7269.545140 8723.350488)
LINESTRING(7228.943036 8691.704220, 7230.263868 8692.301820)
LINESTRING(7230.130016 8690.909669, 7236.635528 8693.947843, 7238.203087 8694.558992, 7246.566879 8698.129852)
LINESTRING(7233.052082 8685.026022, 7233.114331 8684.900731)
LINESTRING(7235.343383 8681.162627, 7235.951754 8680.340183)
LINESTRING(7233.114331 8684.900731, 7234.209751 8682.695150, 7234.367017 8682.488447, 7234.542142 8682.245745, 7235.343383 8681.162627)
LINESTRING(7230.130016 8690.909669, 7233.052082 8685.026022)
LINESTRING(7255.510243 8699.099074, 7250.881096 8693.964737, 7245.560806 8688.592006, 7239.691219 8683.390375, 7235.951754 8680.340183)
LINESTRING(7219.235906 8686.145197, 7230.130016 8690.909669)
LINESTRING(7171.129701 8650.784296, 7172.734677 8650.679301, 7173.495099 8650.929033, 7174.214079 8651.165216, 7192.126998 8660.929323, 7197.898450 8664.075326)
LINESTRING(7150.309569 8645.723660, 7151.809323 8646.372283, 7152.455452 8646.687836)
LINESTRING(7178.488951 8636.167899, 7179.095849 8636.294381)
LINESTRING(7235.951754 8680.340183, 7235.062809 8679.661965, 7231.893165 8677.230009, 7226.680365 8673.607106, 7220.970935 8670.106261, 7215.024189 8666.676907, 7210.789172 8664.323641, 7203.506230 8660.303943, 7191.076082 8653.528460, 7187.602110 8651.634803, 7183.533657 8649.417090, 7179.077877 8646.988365, 7173.680598 8644.046343)
LINESTRING(7173.680598 8644.046343, 7178.431691 8637.105203, 7179.218646 8635.065449)
LINESTRING(7152.922545 8639.584441, 7153.182312 8638.969380)
LINESTRING(7152.922545 8639.584441, 7153.400580 8639.845512, 7155.481039 8640.973587, 7159.220787 8643.001380, 7165.913783 8646.640781, 7168.378110 8647.984063, 7170.192680 8648.973184, 7170.323414 8649.225921, 7170.529550 8649.624302, 7171.129701 8650.784296)
LINESTRING(7149.314835 8641.775055, 7151.010860 8642.887597, 7150.309569 8645.723660)
LINESTRING(7159.331735 8627.200894, 7160.009443 8627.429027, 7160.622293 8627.964718)
LINESTRING(7160.622293 8627.964718, 7161.134457 8628.255496, 7161.850828 8628.826280, 7163.425814 8629.276309, 7164.344580 8629.632567, 7165.696139 8630.103005, 7169.024466 8631.261468, 7171.403754 8632.280239, 7173.416693 8633.310123, 7174.659005 8634.026778, 7175.660031 8634.604195, 7177.199414 8635.492233, 7178.007798 8635.958532, 7178.488951 8636.167899)
LINESTRING(7160.535723 8626.610494, 7160.009443 8627.429027, 7159.280315 8628.578646, 7155.343729 8634.079106)
LINESTRING(7155.846879 8634.352252, 7154.349676 8636.571269, 7153.182312 8638.969380)
LINESTRING(7155.343729 8634.079106, 7156.744044 8634.838564, 7157.181770 8635.075880, 7158.321865 8635.694060, 7164.473783 8639.029814, 7171.316220 8642.757487, 7173.680598 8644.046343)
LINESTRING(7155.846879 8634.352252, 7155.343729 8634.079106)
LINESTRING(7160.535723 8626.610494, 7163.093197 8627.735509, 7169.459698 8630.137304, 7172.826746 8631.579005, 7178.738627 8634.989310, 7179.218646 8635.065449)
LINESTRING(7152.922545 8639.584441, 7146.890306 8636.283666, 7136.488517 8630.585235, 7134.319162 8629.396894, 7124.383899 8624.969065, 7123.323969 8624.777953, 7121.539899 8624.753065, 7118.794885 8624.141065, 7118.036787 8624.066457, 7117.057928 8624.132050, 7116.004913 8624.447093)
LINESTRING(7116.004913 8624.447093, 7112.647899 8624.312050, 7111.216913 8624.249065, 7107.446608 8624.105802, 7101.497594 8623.934816, 7098.702406 8624.024447)
LINESTRING(7097.387528 8625.209953, 7096.202759 8628.525808, 7096.091868 8628.836088, 7094.616094 8632.063616, 7094.575502 8632.152454, 7094.528787 8632.254614, 7094.032838 8633.339263, 7097.047370 8635.481575)
LINESTRING(7098.702406 8624.024447, 7099.318488 8623.211528)
LINESTRING(7097.387528 8625.209953, 7098.702406 8624.024447)
LINESTRING(7089.693165 8635.350614, 7093.515969 8630.246551, 7097.387528 8625.209953)
LINESTRING(7104.226620 8612.107370, 7104.311093 8611.937461)
LINESTRING(7099.318488 8623.211528, 7104.226620 8612.107370)
LINESTRING(7097.887446 8624.084598, 7083.675950 8619.934734, 7065.840416 8611.314803)
LINESTRING(7065.840416 8611.314803, 7065.689216 8610.036491, 7065.330009 8608.768101)
LINESTRING(7065.849090 8609.703591, 7066.134992 8610.118299, 7083.707584 8618.672806, 7085.072580 8619.137802, 7099.318488 8623.211528)
LINESTRING(7065.849090 8609.703591, 7065.689216 8610.036491, 7065.592441 8610.237921)
LINESTRING(7115.152479 8607.998154, 7118.572479 8608.358154, 7119.388517 8608.574154, 7124.440479 8611.886154, 7128.544535 8614.742117, 7129.348498 8615.426117, 7136.590847 8621.963206, 7137.610186 8622.521575, 7138.110047 8622.795345, 7155.354331 8626.387465, 7159.331735 8627.200894)
LINESTRING(7155.343729 8634.079106, 7154.315490 8633.515861, 7153.494236 8633.065890, 7144.602803 8628.194211, 7134.657279 8622.746306, 7128.239131 8619.230608, 7122.178772 8615.910898, 7116.403011 8612.747093, 7112.053701 8610.364630, 7108.643225 8608.496485, 7106.827238 8606.270154)
LINESTRING(7106.625298 8606.725342, 7106.827238 8606.270154)
LINESTRING(7106.625298 8606.725342, 7105.304126 8609.703307, 7104.311093 8611.937461)
LINESTRING(7106.827238 8606.270154, 7109.809455 8599.548132, 7110.782362 8597.356157)
LINESTRING(7079.423924 8595.277739, 7078.770085 8595.916724, 7076.599654 8596.376220)
LINESTRING(7076.599654 8596.376220, 7073.598444 8593.672479)
LINESTRING(7073.598444 8593.672479, 7072.557335 8594.218658, 7071.476598 8594.855376)
LINESTRING(7102.205405 8610.012227, 7095.419206 8606.216466, 7089.858992 8602.878217, 7084.575269 8599.029789, 7079.423924 8595.277739)
LINESTRING(7074.076422 8591.860120, 7073.526784 8591.375282)
LINESTRING(7072.623950 8594.066154, 7072.557335 8594.218658, 7065.849090 8609.703591)
LINESTRING(7073.526784 8591.375282, 7074.141449 8590.748598, 7074.097682 8590.508277)
LINESTRING(7104.226620 8612.107370, 7097.580850 8608.486054, 7094.466028 8606.711339, 7090.380907 8604.284655, 7087.550343 8602.438507, 7083.294236 8599.398803, 7081.232315 8597.852050, 7078.770085 8595.916724, 7074.076422 8591.860120)
LINESTRING(7072.623950 8594.066154, 7073.270646 8592.708359, 7074.076422 8591.860120)
LINESTRING(7076.039187 8586.425254, 7076.445109 8585.496170)
LINESTRING(7073.644989 8591.195792, 7074.097682 8590.508277, 7074.902551 8589.026835, 7076.039187 8586.425254)
LINESTRING(7076.445109 8585.496170, 7078.460542 8580.883238)
LINESTRING(7078.460542 8580.883238, 7088.449323 8586.079937, 7107.773102 8595.942917, 7107.972548 8596.047402, 7110.782362 8597.356157)"""


# =============================================================================
# CORE GEOMETRY ANALYSIS FUNCTIONS
# =============================================================================

def parse_wkt_data(wkt_text: str) -> list:
    """
    Parse WKT text containing multiple LINESTRING geometries.
    Handles multi-line WKT with whitespace/newlines in coordinate lists.
    
    Args:
        wkt_text: Raw WKT text containing LINESTRING definitions
        
    Returns:
        List of Shapely LineString objects
    """
    lines = []
    
    # Clean up the text - normalize whitespace within coordinates
    # This handles the multi-line format in the source file
    cleaned_text = re.sub(r'\s+', ' ', wkt_text)
    
    # Find all LINESTRING definitions
    pattern = r'LINESTRING\s*\([^)]+\)'
    matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
    
    for match in matches:
        try:
            geom = wkt.loads(match)
            if isinstance(geom, LineString) and geom.is_valid:
                lines.append(geom)
        except Exception as e:
            st.warning(f"Could not parse geometry: {str(e)[:50]}")
            continue
    
    return lines


def get_endpoints(line: LineString) -> tuple:
    """
    Extract start and end points from a LineString.
    
    Args:
        line: Shapely LineString object
        
    Returns:
        Tuple of (start_point, end_point) as coordinate tuples
    """
    coords = list(line.coords)
    return (coords[0], coords[-1])


def round_point(point: tuple, precision: int = 6) -> tuple:
    """
    Round point coordinates for consistent comparison.
    Floating point precision issues can cause identical points to appear different.
    """
    return (round(point[0], precision), round(point[1], precision))


def detect_undershoots(lines: list, threshold: float = 5.0) -> list:
    """
    Detect undershoot errors in a road network using dangling node analysis.
    
    Algorithm:
    1. Extract all endpoints from all LineStrings
    2. Count occurrences of each endpoint (point degree)
    3. Identify dangling nodes (degree = 1, only appear once)
    4. For each dangle, find distance to nearest OTHER line segment
    5. If distance > 0 and < threshold, flag as undershoot error
    
    Args:
        lines: List of Shapely LineString objects
        threshold: Maximum distance (in map units) to consider as undershoot
        
    Returns:
        List of dicts containing error details:
        {
            'x': float,          # X coordinate of error
            'y': float,          # Y coordinate of error  
            'distance': float,   # Gap distance to nearest road
            'error_type': str    # Classification of error
        }
    """
    if not lines:
        return []
    
    # Step 1: Extract all endpoints and count occurrences
    endpoint_count = defaultdict(int)
    endpoint_to_line_idx = defaultdict(list)
    
    for idx, line in enumerate(lines):
        start, end = get_endpoints(line)
        start_rounded = round_point(start)
        end_rounded = round_point(end)
        
        endpoint_count[start_rounded] += 1
        endpoint_count[end_rounded] += 1
        endpoint_to_line_idx[start_rounded].append(idx)
        endpoint_to_line_idx[end_rounded].append(idx)
    
    # Step 2: Identify dangling nodes (degree = 1)
    dangles = [pt for pt, count in endpoint_count.items() if count == 1]
    
    # Step 3: For each dangle, check distance to other lines
    errors = []
    
    for dangle in dangles:
        dangle_point = Point(dangle)
        dangle_line_idx = endpoint_to_line_idx[dangle][0]
        
        min_distance = float('inf')
        
        # Check distance to all OTHER lines
        for idx, line in enumerate(lines):
            if idx == dangle_line_idx:
                continue  # Skip the line this dangle belongs to
            
            dist = dangle_point.distance(line)
            if dist < min_distance:
                min_distance = dist
        
        # Step 4: Classify the dangle
        if min_distance > 0 and min_distance < threshold:
            # This is an undershoot - close but not connected
            error_type = "UNDERSHOOT" if min_distance < threshold/2 else "POTENTIAL_GAP"
            errors.append({
                'x': dangle[0],
                'y': dangle[1],
                'distance': round(min_distance, 4),
                'error_type': error_type,
                'severity': 'HIGH' if min_distance < threshold/2 else 'MEDIUM'
            })
    
    return errors


def get_bounds(lines: list) -> tuple:
    """
    Calculate bounding box of all LineStrings.
    
    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    if not lines:
        return (0, 0, 1, 1)
    
    all_x = []
    all_y = []
    
    for line in lines:
        for coord in line.coords:
            all_x.append(coord[0])
            all_y.append(coord[1])
    
    return (min(all_x), min(all_y), max(all_x), max(all_y))


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_map(lines: list, errors: list) -> folium.Map:
    """
    Create an interactive Folium map displaying the road network and errors.
    
    Since the WKT data uses local projected coordinates (not lat/lon),
    we normalize coordinates to a pseudo-geographic range for display.
    
    Args:
        lines: List of Shapely LineString objects
        errors: List of error dictionaries from detect_undershoots()
        
    Returns:
        Folium Map object
    """
    if not lines:
        return folium.Map(location=[0, 0], zoom_start=2)
    
    # Get bounds of the data
    min_x, min_y, max_x, max_y = get_bounds(lines)
    
    # Calculate center and normalize to pseudo lat/lon
    # We'll use a simple linear transformation to fit in a reasonable range
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Scale factor to normalize coordinates
    # Map the local coordinates to a small geographic area
    scale = 0.0001  # Adjust for appropriate zoom level
    
    def normalize_coord(x, y):
        """Convert local coords to pseudo lat/lon for display."""
        lat = (y - center_y) * scale
        lon = (x - center_x) * scale
        return [lat, lon]
    
    # Create map centered on data
    m = folium.Map(
        location=[0, 0],
        zoom_start=15,
        tiles='cartodbpositron'  # Clean, light basemap
    )
    
    # Add road network as grey lines
    for line in lines:
        coords = [normalize_coord(c[0], c[1]) for c in line.coords]
        folium.PolyLine(
            coords,
            color='#666666',
            weight=2,
            opacity=0.6,
            popup='Road Segment'
        ).add_to(m)
    
    # Add error markers in RED
    for error in errors:
        loc = normalize_coord(error['x'], error['y'])
        
        # Color based on severity
        color = '#FF0000' if error['severity'] == 'HIGH' else '#FF6600'
        
        folium.CircleMarker(
            location=loc,
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=folium.Popup(
                f"""<b>🚨 {error['error_type']}</b><br>
                <b>Gap Distance:</b> {error['distance']:.4f} units<br>
                <b>Severity:</b> {error['severity']}<br>
                <b>Coordinates:</b><br>
                X: {error['x']:.2f}<br>
                Y: {error['y']:.2f}""",
                max_width=250
            ),
            tooltip=f"{error['error_type']}: {error['distance']:.4f} units"
        ).add_to(m)
    
    # Fit map to bounds
    if lines:
        all_coords = []
        for line in lines:
            for c in line.coords:
                all_coords.append(normalize_coord(c[0], c[1]))
        m.fit_bounds(all_coords)
    
    return m


# =============================================================================
# STREAMLIT APP INTERFACE
# =============================================================================

def main():
    """Main Streamlit application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Axes Systems: AI Map Validator",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E3A5F;
            text-align: center;
            padding: 1rem 0;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .error-high {
            color: #FF0000;
            font-weight: bold;
        }
        .error-medium {
            color: #FF6600;
            font-weight: bold;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🗺️ Axes Systems: AI Map Validator</p>', unsafe_allow_html=True)
    st.markdown("""
        <p style="text-align: center; color: #666;">
        Automated topology checking for road networks using dangling node detection
        </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar - Data Input
    with st.sidebar:
        st.header("📁 Data Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload WKT File",
            type=['wkt', 'txt'],
            help="Upload a .wkt or .txt file containing LINESTRING geometries"
        )
        
        st.markdown("---")
        
        # Demo data button
        if st.button("🎯 Load Demo Data", type="primary"):
            st.session_state['use_demo'] = True
            st.session_state['uploaded_data'] = None
        
        st.markdown("---")
        
        # Analysis parameters
        st.header("⚙️ Parameters")
        threshold = st.slider(
            "Undershoot Threshold (map units)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Maximum distance to consider as a potential undershoot error"
        )
        
        st.markdown("---")
        
        # Info section
        st.header("ℹ️ About")
        st.markdown("""
        **How it works:**
        1. Parses LINESTRING geometries from WKT
        2. Extracts all road endpoints
        3. Identifies dangling nodes (degree=1)
        4. Flags undershoots where dangles are close to other roads
        
        **Error Types:**
        - 🔴 **UNDERSHOOT**: Gap < 2.5 units (high confidence error)
        - 🟠 **POTENTIAL_GAP**: Gap 2.5-5 units (medium confidence)
        """)
    
    # Main content area
    # Determine data source
    wkt_data = None
    
    if uploaded_file is not None:
        wkt_data = uploaded_file.read().decode('utf-8')
        st.session_state['use_demo'] = False
    elif st.session_state.get('use_demo', False):
        wkt_data = DEMO_WKT_DATA
    
    if wkt_data:
        # Parse and analyze
        with st.spinner("🔍 Analyzing road network..."):
            lines = parse_wkt_data(wkt_data)
            errors = detect_undershoots(lines, threshold=threshold)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Roads",
                value=len(lines),
                help="Number of LineString geometries processed"
            )
        
        with col2:
            st.metric(
                label="🚨 Anomalies Detected",
                value=len(errors),
                delta=f"{len(errors)} issues found" if errors else "Clean!",
                delta_color="inverse" if errors else "normal"
            )
        
        with col3:
            high_severity = sum(1 for e in errors if e['severity'] == 'HIGH')
            st.metric(
                label="🔴 High Severity",
                value=high_severity,
                help="Undershoots with gap < 2.5 units"
            )
        
        with col4:
            medium_severity = sum(1 for e in errors if e['severity'] == 'MEDIUM')
            st.metric(
                label="🟠 Medium Severity",
                value=medium_severity,
                help="Potential gaps (2.5-5 units)"
            )
        
        st.divider()
        
        # Map visualization
        st.subheader("🗺️ Interactive Map")
        st.caption("Grey lines = Road network | Red/Orange markers = Detected errors (click for details)")
        
        map_obj = create_map(lines, errors)
        st_folium(map_obj, width="stretch", height=500)
        
        st.divider()
        
        # Error details table
        if errors:
            st.subheader("📋 Error Details")
            
            df = pd.DataFrame(errors)
            df = df.rename(columns={
                'x': 'X Coordinate',
                'y': 'Y Coordinate', 
                'distance': 'Gap Distance',
                'error_type': 'Error Type',
                'severity': 'Severity'
            })
            
            # Style the dataframe
            def highlight_severity(val):
                if val == 'HIGH':
                    return 'background-color: #ffcccc'
                elif val == 'MEDIUM':
                    return 'background-color: #fff3cd'
                return ''
            
            styled_df = df.style.map(
                highlight_severity, 
                subset=['Severity']
            )
            
            st.dataframe(
                styled_df,
                width="stretch",
                hide_index=True
            )
            
            # Download button for errors
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Error Report (CSV)",
                data=csv,
                file_name="map_validation_errors.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No topology errors detected! The road network appears to be clean.")
    
    else:
        # No data loaded - show instructions
        st.info("""
        👈 **Get started by:**
        1. Uploading a WKT file using the sidebar, OR
        2. Clicking **"Load Demo Data"** to see the validator in action
        
        The validator will analyze your road network and detect undershoots, 
        gaps, and other topology errors automatically.
        """)
        
        # Show example of expected format
        with st.expander("📖 Expected WKT Format"):
            st.code("""
LINESTRING(x1 y1, x2 y2, x3 y3, ...)
LINESTRING(x1 y1, x2 y2, ...)
...
            """, language="text")
            st.markdown("""
            Each line should be a valid WKT LINESTRING geometry.
            Coordinates can be in any projected coordinate system.
            """)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
