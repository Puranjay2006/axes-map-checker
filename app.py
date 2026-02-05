"""
Axes Systems: AI Map Validator
==============================
A professional Streamlit web application for detecting topology errors in road networks.
Uses advanced dangling node detection and spatial analysis algorithms.

Author: Axes Systems Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely import wkt
from shapely.geometry import Point, LineString
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_CONFIG = {
    "title": "Axes Systems: AI Map Validator",
    "version": "2.0.0",
    "icon": "🗺️",
    "default_threshold": 5.0,
    "precision": 6,
}

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
# GEOMETRY ANALYSIS ENGINE
# =============================================================================

class GeometryAnalyzer:
    """Core geometry analysis engine for topology validation."""
    
    def __init__(self, precision: int = 6):
        self.precision = precision
        self.lines: List[LineString] = []
        self.endpoint_index: Dict[Tuple, List[int]] = defaultdict(list)
        self.endpoint_count: Dict[Tuple, int] = defaultdict(int)
    
    def parse_wkt(self, wkt_text: str) -> List[LineString]:
        """Parse WKT text containing LINESTRING geometries."""
        self.lines = []
        cleaned_text = re.sub(r'\s+', ' ', wkt_text)
        pattern = r'LINESTRING\s*\([^)]+\)'
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        
        for match in matches:
            try:
                geom = wkt.loads(match)
                if isinstance(geom, LineString) and geom.is_valid and not geom.is_empty:
                    self.lines.append(geom)
            except Exception:
                continue
        
        self._build_index()
        return self.lines
    
    def _build_index(self) -> None:
        """Build spatial index of endpoints."""
        self.endpoint_index.clear()
        self.endpoint_count.clear()
        
        for idx, line in enumerate(self.lines):
            coords = list(line.coords)
            start = self._round_point(coords[0])
            end = self._round_point(coords[-1])
            
            self.endpoint_count[start] += 1
            self.endpoint_count[end] += 1
            self.endpoint_index[start].append(idx)
            self.endpoint_index[end].append(idx)
    
    def _round_point(self, point: Tuple) -> Tuple:
        """Round coordinates for comparison."""
        return (round(point[0], self.precision), round(point[1], self.precision))
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of all geometries."""
        if not self.lines:
            return (0, 0, 1, 1)
        
        all_x, all_y = [], []
        for line in self.lines:
            for coord in line.coords:
                all_x.append(coord[0])
                all_y.append(coord[1])
        
        return (min(all_x), min(all_y), max(all_x), max(all_y))
    
    def get_total_length(self) -> float:
        """Calculate total network length."""
        return sum(line.length for line in self.lines)
    
    def detect_undershoots(self, threshold: float = 5.0) -> List[Dict]:
        """Detect undershoot errors using dangling node analysis."""
        errors = []
        dangles = [pt for pt, count in self.endpoint_count.items() if count == 1]
        
        for dangle in dangles:
            dangle_point = Point(dangle)
            dangle_line_idx = self.endpoint_index[dangle][0]
            
            min_distance = float('inf')
            nearest_line_idx = -1
            
            for idx, line in enumerate(self.lines):
                if idx == dangle_line_idx:
                    continue
                dist = dangle_point.distance(line)
                if dist < min_distance:
                    min_distance = dist
                    nearest_line_idx = idx
            
            if 0 < min_distance < threshold:
                severity = 'HIGH' if min_distance < threshold / 2 else 'MEDIUM'
                error_type = 'UNDERSHOOT' if severity == 'HIGH' else 'POTENTIAL_GAP'
                
                errors.append({
                    'x': dangle[0],
                    'y': dangle[1],
                    'distance': round(min_distance, 4),
                    'error_type': error_type,
                    'severity': severity,
                    'source_line': dangle_line_idx,
                    'nearest_line': nearest_line_idx
                })
        
        return errors
    
    def detect_dead_ends(self) -> List[Dict]:
        """Detect valid dead ends (endpoints far from other roads)."""
        dead_ends = []
        dangles = [pt for pt, count in self.endpoint_count.items() if count == 1]
        
        for dangle in dangles:
            dangle_point = Point(dangle)
            dangle_line_idx = self.endpoint_index[dangle][0]
            
            min_distance = float('inf')
            for idx, line in enumerate(self.lines):
                if idx == dangle_line_idx:
                    continue
                dist = dangle_point.distance(line)
                if dist < min_distance:
                    min_distance = dist
            
            if min_distance >= 5.0:
                dead_ends.append({
                    'x': dangle[0],
                    'y': dangle[1],
                    'nearest_distance': round(min_distance, 4)
                })
        
        return dead_ends
    
    def get_statistics(self) -> Dict:
        """Generate network statistics."""
        if not self.lines:
            return {}
        
        lengths = [line.length for line in self.lines]
        
        return {
            'total_segments': len(self.lines),
            'total_length': round(sum(lengths), 2),
            'avg_length': round(sum(lengths) / len(lengths), 2),
            'min_length': round(min(lengths), 2),
            'max_length': round(max(lengths), 2),
            'total_endpoints': len(self.endpoint_count),
            'connected_nodes': sum(1 for c in self.endpoint_count.values() if c > 1),
            'dangling_nodes': sum(1 for c in self.endpoint_count.values() if c == 1)
        }


# =============================================================================
# MAP VISUALIZATION
# =============================================================================

def create_enhanced_map(
    analyzer: GeometryAnalyzer,
    errors: List[Dict],
    show_dead_ends: bool = False,
    dead_ends: Optional[List[Dict]] = None
) -> folium.Map:
    """Create an enhanced interactive Folium map."""
    
    if not analyzer.lines:
        return folium.Map(location=[0, 0], zoom_start=2, tiles='cartodbpositron')
    
    min_x, min_y, max_x, max_y = analyzer.get_bounds()
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    scale = 0.0001
    
    def normalize(x: float, y: float) -> List[float]:
        return [(y - center_y) * scale, (x - center_x) * scale]
    
    # Create base map
    m = folium.Map(
        location=[0, 0],
        zoom_start=15,
        tiles='cartodbpositron'
    )
    
    # Road network layer
    road_group = folium.FeatureGroup(name='Road Network')
    for i, line in enumerate(analyzer.lines):
        coords = [normalize(c[0], c[1]) for c in line.coords]
        folium.PolyLine(
            coords,
            color='#4a4a4a',
            weight=2.5,
            opacity=0.7,
            popup=f"<b>Road Segment #{i+1}</b><br>Length: {line.length:.2f} units"
        ).add_to(road_group)
    road_group.add_to(m)
    
    # Error markers layer
    if errors:
        error_group = folium.FeatureGroup(name='Detected Errors')
        for i, error in enumerate(errors):
            loc = normalize(error['x'], error['y'])
            color = '#dc3545' if error['severity'] == 'HIGH' else '#fd7e14'
            
            folium.CircleMarker(
                location=loc,
                radius=10,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.85,
                popup=folium.Popup(
                    f"""<div style="font-family: Arial; min-width: 180px;">
                    <h4 style="color: {color}; margin: 0 0 8px 0;">⚠️ {error['error_type']}</h4>
                    <table style="width: 100%;">
                        <tr><td><b>Gap:</b></td><td>{error['distance']:.4f} units</td></tr>
                        <tr><td><b>Severity:</b></td><td>{error['severity']}</td></tr>
                        <tr><td><b>X:</b></td><td>{error['x']:.2f}</td></tr>
                        <tr><td><b>Y:</b></td><td>{error['y']:.2f}</td></tr>
                    </table>
                    </div>""",
                    max_width=220
                ),
                tooltip=f"Error #{i+1}: {error['error_type']}"
            ).add_to(error_group)
        error_group.add_to(m)
    
    # Dead ends layer (optional)
    if show_dead_ends and dead_ends:
        dead_end_group = folium.FeatureGroup(name='Dead Ends', show=False)
        for de in dead_ends:
            loc = normalize(de['x'], de['y'])
            folium.CircleMarker(
                location=loc,
                radius=6,
                color='#6c757d',
                fill=True,
                fillColor='#6c757d',
                fillOpacity=0.6,
                tooltip=f"Dead End (nearest: {de['nearest_distance']:.1f})"
            ).add_to(dead_end_group)
        dead_end_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Fit bounds
    all_coords = [normalize(c[0], c[1]) for line in analyzer.lines for c in line.coords]
    m.fit_bounds(all_coords)
    
    return m


# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

def render_metrics_dashboard(stats: Dict, errors: List[Dict]) -> None:
    """Render the metrics dashboard."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Segments",
            value=stats.get('total_segments', 0),
            help="Number of road segments in the network"
        )
    
    with col2:
        error_count = len(errors)
        st.metric(
            label="Anomalies Found",
            value=error_count,
            delta="Clean!" if error_count == 0 else f"{error_count} issues",
            delta_color="normal" if error_count == 0 else "inverse"
        )
    
    with col3:
        high_count = sum(1 for e in errors if e['severity'] == 'HIGH')
        st.metric(
            label="High Severity",
            value=high_count,
            help="Critical undershoots requiring attention"
        )
    
    with col4:
        medium_count = sum(1 for e in errors if e['severity'] == 'MEDIUM')
        st.metric(
            label="Medium Severity",
            value=medium_count,
            help="Potential gaps for review"
        )


def render_statistics_panel(stats: Dict) -> None:
    """Render detailed statistics panel."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Network Metrics**")
        st.write(f"- Total Length: **{stats.get('total_length', 0):,.2f}** units")
        st.write(f"- Avg Segment: **{stats.get('avg_length', 0):.2f}** units")
        st.write(f"- Min/Max: **{stats.get('min_length', 0):.2f}** / **{stats.get('max_length', 0):.2f}** units")
    
    with col2:
        st.markdown("**Topology Metrics**")
        st.write(f"- Total Nodes: **{stats.get('total_endpoints', 0)}**")
        st.write(f"- Connected: **{stats.get('connected_nodes', 0)}**")
        st.write(f"- Dangling: **{stats.get('dangling_nodes', 0)}**")


def render_error_table(errors: List[Dict]) -> None:
    """Render the error details table."""
    
    if not errors:
        st.success("✅ No topology errors detected! The network is clean.")
        return
    
    df = pd.DataFrame(errors)
    df = df[['error_type', 'severity', 'distance', 'x', 'y']]
    df.columns = ['Error Type', 'Severity', 'Gap Distance', 'X Coordinate', 'Y Coordinate']
    df.index = df.index + 1
    df.index.name = '#'
    
    # Apply styling
    def style_severity(val):
        if val == 'HIGH':
            return 'background-color: #f8d7da; color: #721c24;'
        elif val == 'MEDIUM':
            return 'background-color: #fff3cd; color: #856404;'
        return ''
    
    styled = df.style.map(style_severity, subset=['Severity'])
    styled = styled.format({
        'Gap Distance': '{:.4f}',
        'X Coordinate': '{:.2f}',
        'Y Coordinate': '{:.2f}'
    })
    
    st.dataframe(styled, use_container_width=True)


def render_sidebar() -> Tuple[Optional[str], float, bool]:
    """Render sidebar and return user inputs."""
    
    with st.sidebar:
        st.title("🗺️ AI Map Validator")
        st.caption(f"v{APP_CONFIG['version']}")
        
        st.divider()
        
        # Data Input Section
        st.subheader("📁 Data Input")
        
        uploaded_file = st.file_uploader(
            "Upload WKT/TXT file",
            type=['wkt', 'txt'],
            help="Upload a file containing LINESTRING geometries"
        )
        
        use_demo = st.button(
            "🎯 Load Demo Data",
            type="primary",
            use_container_width=True
        )
        
        st.divider()
        
        # Parameters Section
        st.subheader("⚙️ Parameters")
        
        threshold = st.slider(
            "Detection Threshold",
            min_value=1.0,
            max_value=20.0,
            value=APP_CONFIG['default_threshold'],
            step=0.5,
            help="Maximum gap distance to flag as error"
        )
        
        show_dead_ends = st.checkbox(
            "Show Dead Ends",
            value=False,
            help="Display legitimate dead ends on map"
        )
        
        st.divider()
        
        # Info Section
        st.subheader("ℹ️ How It Works")
        st.markdown("""
        1. **Parse** geometry data
        2. **Index** all endpoints  
        3. **Identify** dangling nodes
        4. **Measure** gaps to nearest roads
        5. **Classify** errors by severity
        """)
        
        st.divider()
        
        st.caption("Built for IIT Mandi Hackathon 3.0")
        st.caption("© 2026 Axes Systems")
    
    # Determine data source
    wkt_data = None
    if use_demo:
        wkt_data = DEMO_WKT_DATA
        st.session_state['data_source'] = 'demo'
    elif uploaded_file is not None:
        wkt_data = uploaded_file.read().decode('utf-8')
        st.session_state['data_source'] = 'upload'
    elif st.session_state.get('data_source') == 'demo':
        wkt_data = DEMO_WKT_DATA
    
    return wkt_data, threshold, show_dead_ends


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Page config
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon=APP_CONFIG['icon'],
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #6c757d;
            margin-bottom: 1.5rem;
        }
        .stMetric {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1rem;
            border-radius: 0.75rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = None
    
    # Render sidebar and get inputs
    wkt_data, threshold, show_dead_ends = render_sidebar()
    
    # Header
    st.markdown('<h1 class="main-header">🗺️ AI Map Validator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated topology validation for road networks</p>', unsafe_allow_html=True)
    
    if wkt_data:
        # Initialize analyzer
        analyzer = GeometryAnalyzer(precision=APP_CONFIG['precision'])
        
        # Parse and analyze
        with st.spinner("Analyzing road network..."):
            analyzer.parse_wkt(wkt_data)
            errors = analyzer.detect_undershoots(threshold)
            stats = analyzer.get_statistics()
            dead_ends = analyzer.detect_dead_ends() if show_dead_ends else None
        
        # Metrics Dashboard
        render_metrics_dashboard(stats, errors)
        
        st.divider()
        
        # Create tabs for organized content
        tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 Error Details", "📊 Statistics"])
        
        with tab1:
            st.subheader("Interactive Network Map")
            st.caption("Click markers for error details • Grey = roads • Red/Orange = errors")
            
            map_obj = create_enhanced_map(analyzer, errors, show_dead_ends, dead_ends)
            st_folium(map_obj, height=550, use_container_width=True)
        
        with tab2:
            st.subheader("Detected Anomalies")
            render_error_table(errors)
            
            if errors:
                # Export button
                df = pd.DataFrame(errors)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Report (CSV)",
                    data=csv_data,
                    file_name="topology_errors_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with tab3:
            st.subheader("Network Statistics")
            render_statistics_panel(stats)
            
            # Quality Score
            if stats.get('total_segments', 0) > 0:
                error_rate = len(errors) / stats['total_segments'] * 100
                quality_score = max(0, 100 - error_rate * 10)
                
                st.divider()
                st.markdown("**Network Quality Score**")
                
                if quality_score >= 90:
                    st.success(f"🌟 Excellent: {quality_score:.1f}%")
                elif quality_score >= 70:
                    st.warning(f"⚠️ Good: {quality_score:.1f}%")
                else:
                    st.error(f"🔴 Needs Attention: {quality_score:.1f}%")
    
    else:
        # Welcome screen
        st.info("""
        **Welcome to AI Map Validator!**
        
        Get started by:
        1. **Upload** a WKT file using the sidebar, or
        2. Click **"Load Demo Data"** to see the validator in action
        
        The system will automatically detect topology errors like undershoots and gaps.
        """)
        
        with st.expander("📖 Supported Format"):
            st.code("""
LINESTRING(x1 y1, x2 y2, x3 y3, ...)
LINESTRING(x1 y1, x2 y2, ...)
            """, language="text")
            st.markdown("Each line should be a valid WKT LINESTRING geometry.")


if __name__ == "__main__":
    main()
