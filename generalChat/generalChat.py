"""
General Chat Context Generator and Service
Analyzes CSV datasets and creates context for LLM-based general chat
"""

import os
import pandas as pd
import json
from typing import Dict, List, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GeneralChatContextGenerator:
    """Generates context from CSV datasets for general chat functionality"""

    def __init__(self, csv_output_dir: str = "datasets/csv_output"):
        self.csv_output_dir = csv_output_dir
        self.column_descriptions = {
            "S.No": "Serial number for entry identification",
            "STATE": "Name of the Indian state",
            "DISTRICT": "Name of the district within a state",
            "ASSESSMENT UNIT": "Smallest administrative or hydrological unit assessed for groundwater estimation",
            "Rainfall (mm)_C": "Rainfall (in mm) in command areas (irrigated by canals)",
            "Rainfall (mm)_NC": "Rainfall (in mm) in non-command areas (not covered by canal irrigation)",
            "Rainfall (mm)_PQ": "Rainfall (in mm) in poor quality areas (highly saline or otherwise unfit for conventional use)",
            "Rainfall (mm)_Total": "Total rainfall (in mm) for the assessment unit (sum of categories)",
            "Total Geographical Area (ha)_Recharge Worthy Area (ha)_C": "Area (in hectares) within command areas suitable for groundwater recharge",
            "Total Geographical Area (ha)_Recharge Worthy Area (ha)_NC": "Area (ha) in non-command suitable for recharge",
            "Total Geographical Area (ha)_Recharge Worthy Area (ha)_PQ": "Area (ha) in poor quality regions suitable for recharge",
            "Total Geographical Area (ha)_Recharge Worthy Area (ha)_Total": "Total area suitable for recharge",
            "Total Geographical Area (ha)_Hilly Area": "Area classified as hilly and less recharge-worthy",
            "Total Geographical Area (ha)_Total": "Full area (ha) covered by the assessment unit or region",
            "Ground Water Recharge (ham)_Rainfall Recharge_C": "Groundwater recharge attributed to rainfall in command areas (in hectare-meters)",
            "Ground Water Recharge (ham)_Rainfall Recharge_NC": "Groundwater recharge attributed to rainfall in non-command areas",
            "Ground Water Recharge (ham)_Rainfall Recharge_PQ": "Groundwater recharge attributed to rainfall in poor quality areas",
            "Ground Water Recharge (ham)_Rainfall Recharge_Total": "Total groundwater recharge from rainfall",
            "Ground Water Recharge (ham)_Canals_C": "Recharge contribution from canals in command areas",
            "Ground Water Recharge (ham)_Canals_NC": "Recharge contribution from canals in non-command areas",
            "Ground Water Recharge (ham)_Canals_PQ": "Recharge contribution from canals in poor quality areas",
            "Ground Water Recharge (ham)_Canals_Total": "Total recharge from canals",
            "Ground Water Recharge (ham)_Surface Water Irrigation_C": "Recharge from surface water irrigation in command areas",
            "Ground Water Recharge (ham)_Surface Water Irrigation_NC": "Recharge from surface water irrigation in non-command areas",
            "Ground Water Recharge (ham)_Surface Water Irrigation_PQ": "Recharge from surface water irrigation in poor quality areas",
            "Ground Water Recharge (ham)_Surface Water Irrigation_Total": "Total surface water irrigation recharge",
            "Ground Water Recharge (ham)_Ground Water Irrigation_C": "Recharge resulting from groundwater irrigation return flow in command areas",
            "Ground Water Recharge (ham)_Ground Water Irrigation_NC": "Recharge resulting from groundwater irrigation return flow in non-command areas",
            "Ground Water Recharge (ham)_Ground Water Irrigation_PQ": "Recharge resulting from groundwater irrigation return flow in poor quality areas",
            "Ground Water Recharge (ham)_Ground Water Irrigation_Total": "Total recharge by groundwater irrigation",
            "Ground Water Recharge (ham)_Tanks and Ponds_C": "Recharge from water bodies like tanks and ponds in command areas",
            "Ground Water Recharge (ham)_Tanks and Ponds_NC": "Recharge from tanks and ponds in non-command areas",
            "Ground Water Recharge (ham)_Tanks and Ponds_PQ": "Recharge from tanks and ponds in poor quality areas",
            "Ground Water Recharge (ham)_Tanks and Ponds_Total": "Total recharge from tanks and ponds",
            "Ground Water Recharge (ham)_Water Conservation Structure_C": "Recharge from artificial structures for water conservation in command areas",
            "Ground Water Recharge (ham)_Water Conservation Structure_NC": "Recharge from water conservation structures in non-command areas",
            "Ground Water Recharge (ham)_Water Conservation Structure_PQ": "Recharge from water conservation structures in poor quality areas",
            "Ground Water Recharge (ham)_Water Conservation Structure_Total": "Total recharge from all water conservation structures",
            "Ground Water Recharge (ham)_Pipelines_C": "Recharge from seepage or leaks from pipelines in command areas",
            "Ground Water Recharge (ham)_Pipelines_NC": "Recharge from pipelines in non-command areas",
            "Ground Water Recharge (ham)_Pipelines_PQ": "Recharge from pipelines in poor quality areas",
            "Ground Water Recharge (ham)_Pipelines_Total": "Total recharge from pipelines",
            "Ground Water Recharge (ham)_Sewages and Flash Flood Channels_C": "Recharge from urban runoff and storm water channels in command areas",
            "Ground Water Recharge (ham)_Sewages and Flash Flood Channels_NC": "Recharge from sewage and flash flood channels in non-command areas",
            "Ground Water Recharge (ham)_Sewages and Flash Flood Channels_PQ": "Recharge from sewage and flood channels in poor quality areas",
            "Ground Water Recharge (ham)_Sewages and Flash Flood Channels_Total": "Total recharge from sewage and flash flood channels",
            "Ground Water Recharge (ham)_C": "Aggregated groundwater recharge in command areas",
            "Ground Water Recharge (ham)_NC": "Aggregated groundwater recharge in non-command areas",
            "Ground Water Recharge (ham)_PQ": "Aggregated groundwater recharge in poor quality areas",
            "Ground Water Recharge (ham)_Total": "Total aggregated groundwater recharge",
            "Inflows and Outflows (ham)_Base Flow_C": "Groundwater flowing out to streams (base flow) in command areas",
            "Inflows and Outflows (ham)_Base Flow_NC": "Base flow to streams in non-command areas",
            "Inflows and Outflows (ham)_Base Flow_PQ": "Base flow to streams in poor quality areas",
            "Inflows and Outflows (ham)_Base Flow_Total": "Total base flow in all areas",
            "Inflows and Outflows (ham)_Stream Recharges_C": "Stream contributions to groundwater recharge in command areas",
            "Inflows and Outflows (ham)_Stream Recharges_NC": "Stream recharge in non-command areas",
            "Inflows and Outflows (ham)_Stream Recharges_PQ": "Stream recharge in poor quality areas",
            "Inflows and Outflows (ham)_Stream Recharges_Total": "Total recharge from streams",
            "Inflows and Outflows (ham)_Lateral Flows_C": "Groundwater flow from adjacent units in command areas",
            "Inflows and Outflows (ham)_Lateral Flows_NC": "Lateral groundwater flows in non-command areas",
            "Inflows and Outflows (ham)_Lateral Flows_PQ": "Lateral flows in poor quality areas",
            "Inflows and Outflows (ham)_Lateral Flows_Total": "Total lateral groundwater flows",
            "Inflows and Outflows (ham)_Vertical Flows_C": "Groundwater exchanged with deeper aquifers (vertical movement) in command areas",
            "Inflows and Outflows (ham)_Vertical Flows_NC": "Vertical groundwater flows in non-command areas",
            "Inflows and Outflows (ham)_Vertical Flows_PQ": "Vertical groundwater flows in poor quality areas",
            "Inflows and Outflows (ham)_Vertical Flows_Total": "Total vertical groundwater flows",
            "Inflows and Outflows (ham)_Evaporation_C": "Groundwater loss via evaporation in command areas",
            "Inflows and Outflows (ham)_Evaporation_NC": "Evaporation loss in non-command areas",
            "Inflows and Outflows (ham)_Evaporation_PQ": "Evaporation loss in poor quality areas",
            "Inflows and Outflows (ham)_Evaporation_Total": "Total groundwater lost by evaporation",
            "Inflows and Outflows (ham)_Transpiration_C": "Groundwater loss by plant transpiration in command areas",
            "Inflows and Outflows (ham)_Transpiration_NC": "Transpiration loss in non-command areas",
            "Inflows and Outflows (ham)_Transpiration_PQ": "Transpiration loss in poor quality areas",
            "Inflows and Outflows (ham)_Transpiration_Total": "Total groundwater lost by transpiration",
            "Inflows and Outflows (ham)_Evapotranspiration_C": "Total water lost to air through evaporation and transpiration in command areas",
            "Inflows and Outflows (ham)_Evapotranspiration_NC": "Evapotranspiration in non-command areas",
            "Inflows and Outflows (ham)_Evapotranspiration_PQ": "Evapotranspiration in poor quality areas",
            "Inflows and Outflows (ham)_Evapotranspiration_Total": "Total groundwater lost by evapotranspiration",
            "Inflows and Outflows (ham)_C": "Net inflows and outflows in command areas",
            "Inflows and Outflows (ham)_NC": "Net inflows and outflows in non-command areas",
            "Inflows and Outflows (ham)_PQ": "Net inflows and outflows in poor quality areas",
            "Inflows and Outflows (ham)_Total": "Total net inflows and outflows",
            "Annual Ground water Recharge (ham)_C": "Computed annual groundwater recharge in command areas",
            "Annual Ground water Recharge (ham)_NC": "Computed annual groundwater recharge in non-command areas",
            "Annual Ground water Recharge (ham)_PQ": "Computed annual groundwater recharge in poor quality areas",
            "Annual Ground water Recharge (ham)_Total": "Total computed annual groundwater recharge",
            "Environmental Flows (ham)_C": "Minimum flows preserved for ecological needs in command areas",
            "Environmental Flows (ham)_NC": "Environmental flows in non-command areas",
            "Environmental Flows (ham)_PQ": "Environmental flows in poor quality areas",
            "Environmental Flows (ham)_Total": "Total environmental flows",
            "Annual Extractable Ground water Resource (ham)_C": "Groundwater available for withdrawal in command areas",
            "Annual Extractable Ground water Resource (ham)_NC": "Annual extractable groundwater in non-command areas",
            "Annual Extractable Ground water Resource (ham)_PQ": "Annual extractable groundwater in poor quality areas",
            "Annual Extractable Ground water Resource (ham)_Total": "Total extractable groundwater available for withdrawal"
        }

    def analyze_csv_files(self) -> Dict[str, Any]:
        """Analyze all CSV files and extract key insights"""
        analysis = {
            "dataset_overview": {},
            "temporal_coverage": [],
            "geographical_coverage": {},
            "key_metrics": {},
            "data_quality": {},
            "summary_statistics": {}
        }

        csv_files = []
        csv_dir = Path(self.csv_output_dir)

        if csv_dir.exists():
            csv_files = [f for f in csv_dir.glob("*.csv") if not f.name.endswith("_headers.json")]

        logger.info(f"Found {len(csv_files)} CSV files to analyze")

        for csv_file in csv_files:
            try:
                year = csv_file.stem  # e.g., "2024-2025"
                analysis["temporal_coverage"].append(year)

                df = pd.read_csv(csv_file)

                # Basic dataset info
                analysis["dataset_overview"][year] = {
                    "total_records": len(df),
                    "total_columns": len(df.columns),
                    "states_covered": df['STATE'].nunique() if 'STATE' in df.columns else 0,
                    "districts_covered": df['DISTRICT'].nunique() if 'DISTRICT' in df.columns else 0
                }

                # Geographical coverage
                if 'STATE' in df.columns:
                    # Filter out NaN values and get unique states
                    states = df['STATE'].dropna().unique().tolist()
                    analysis["geographical_coverage"][year] = states

                # Key metrics analysis
                key_metrics = {}

                # Focus on important metrics
                important_metrics = [
                    'Rainfall (mm)_Total',
                    'Annual Ground water Recharge (ham)_Total',
                    'Annual Extractable Ground water Resource (ham)_Total',
                    'Ground Water Extraction for all uses (ha.m)_Total',
                    'Stage of Ground Water Extraction (%)_Total'
                ]

                for metric in important_metrics:
                    if metric in df.columns:
                        # Convert to numeric, handling any non-numeric values
                        numeric_series = pd.to_numeric(df[metric], errors='coerce')
                        key_metrics[metric] = {
                            "mean": float(numeric_series.mean()) if pd.notna(numeric_series.mean()) else 0,
                            "median": float(numeric_series.median()) if pd.notna(numeric_series.median()) else 0,
                            "min": float(numeric_series.min()) if pd.notna(numeric_series.min()) else 0,
                            "max": float(numeric_series.max()) if pd.notna(numeric_series.max()) else 0,
                            "std": float(numeric_series.std()) if pd.notna(numeric_series.std()) else 0
                        }

                analysis["key_metrics"][year] = key_metrics

                # Data quality assessment
                analysis["data_quality"][year] = {
                    "missing_values_percentage": float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                    "complete_records": int(df.dropna().shape[0]),
                    "total_records": int(len(df))
                }

            except Exception as e:
                logger.error(f"Error analyzing {csv_file}: {e}")
                continue

        return analysis

    def generate_context_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive context summary under 4000 tokens"""

        context_parts = []

        # Dataset Overview
        context_parts.append("# Indian Groundwater Resource Assessment Dataset")
        context_parts.append("\n## Dataset Overview")
        context_parts.append("This dataset contains comprehensive groundwater resource assessment data for India, covering multiple years and providing detailed information about groundwater recharge, extraction, availability, and quality across different states and districts.")

        # Temporal Coverage
        if analysis["temporal_coverage"]:
            # Convert to strings and sort
            years = sorted([str(year) for year in analysis["temporal_coverage"]])
            context_parts.append(f"\n## Temporal Coverage")
            context_parts.append(f"Data available for years: {', '.join(years)}")
            context_parts.append(f"Total years covered: {len(years)}")

        # Geographical Coverage
        if analysis["geographical_coverage"]:
            all_states = set()
            for year_states in analysis["geographical_coverage"].values():
                # Filter out NaN values and convert to strings
                valid_states = [str(state) for state in year_states if pd.notna(state) and str(state).strip()]
                all_states.update(valid_states)

            context_parts.append(f"\n## Geographical Coverage")
            context_parts.append(f"Total states covered: {len(all_states)}")
            if all_states:
                sorted_states = sorted(list(all_states))
                context_parts.append(f"States include: {', '.join(sorted_states[:10])}{'...' if len(all_states) > 10 else ''}")

        # Dataset Structure and Key Metrics
        context_parts.append("\n## Key Data Categories")
        context_parts.append("The dataset is organized into several key categories:")

        categories = [
            ("Rainfall Data", "Rainfall measurements in mm across command areas (C), non-command areas (NC), and poor quality areas (PQ)"),
            ("Geographical Areas", "Total geographical area, recharge-worthy areas, and hilly areas in hectares"),
            ("Groundwater Recharge", "Recharge from various sources including rainfall, canals, surface water irrigation, groundwater irrigation, tanks/ponds, water conservation structures, pipelines, and sewage/flood channels"),
            ("Inflows and Outflows", "Base flow, stream recharges, lateral flows, vertical flows, evaporation, transpiration, and evapotranspiration"),
            ("Annual Recharge and Resources", "Annual groundwater recharge, environmental flows, and extractable groundwater resources"),
            ("Groundwater Extraction", "Extraction for domestic, industrial, and irrigation uses"),
            ("Extraction Stages", "Percentage of extractable resource being withdrawn"),
            ("Future Projections", "Allocation for domestic use in 2025 and net availability for future use"),
            ("Quality Assessment", "Quality tagging for major and other parameters"),
            ("Special Conditions", "Additional potential resources under specific conditions like waterlogging, flood-prone areas, and spring discharge"),
            ("Aquifer Types", "Data for unconfined, confined, and semi-confined aquifers with fresh and saline water classification")
        ]

        for category, description in categories:
            context_parts.append(f"- **{category}**: {description}")

        # Area Classifications
        context_parts.append("\n## Area Classifications")
        context_parts.append("Data is categorized by area types:")
        context_parts.append("- **Command Areas (C)**: Areas irrigated by canals")
        context_parts.append("- **Non-Command Areas (NC)**: Areas not covered by canal irrigation")
        context_parts.append("- **Poor Quality Areas (PQ)**: Areas with highly saline or otherwise unfit water for conventional use")

        # Units and Measurements
        context_parts.append("\n## Units and Measurements")
        context_parts.append("- **mm**: Millimeters (for rainfall)")
        context_parts.append("- **ha**: Hectares (for area measurements)")
        context_parts.append("- **ham**: Hectare-meters (for groundwater volumes)")
        context_parts.append("- **%**: Percentage (for extraction stages)")

        # Key Insights from Analysis
        if analysis["key_metrics"]:
            context_parts.append("\n## Key Insights")

            # Get latest year data for insights
            latest_year = max([str(year) for year in analysis["temporal_coverage"]]) if analysis["temporal_coverage"] else None
            if latest_year and latest_year in analysis["key_metrics"]:
                metrics = analysis["key_metrics"][latest_year]

                if "Annual Extractable Ground water Resource (ham)_Total" in metrics:
                    avg_extractable = metrics["Annual Extractable Ground water Resource (ham)_Total"]["mean"]
                    context_parts.append(f"- Average annual extractable groundwater resource: {avg_extractable:.2f} hectare-meters")

                if "Stage of Ground Water Extraction (%)_Total" in metrics:
                    avg_extraction_stage = metrics["Stage of Ground Water Extraction (%)_Total"]["mean"]
                    context_parts.append(f"- Average groundwater extraction stage: {avg_extraction_stage:.2f}%")

                if "Rainfall (mm)_Total" in metrics:
                    avg_rainfall = metrics["Rainfall (mm)_Total"]["mean"]
                    context_parts.append(f"- Average annual rainfall: {avg_rainfall:.2f} mm")

        # Data Quality Information
        if analysis["data_quality"]:
            context_parts.append("\n## Data Quality")
            total_records = sum(year_data["total_records"] for year_data in analysis["data_quality"].values())
            avg_completeness = sum(year_data["missing_values_percentage"] for year_data in analysis["data_quality"].values()) / len(analysis["data_quality"])
            context_parts.append(f"- Total records across all years: {total_records:,}")
            context_parts.append(f"- Average data completeness: {100 - avg_completeness:.1f}%")

        # Usage Guidelines
        context_parts.append("\n## Usage Guidelines")
        context_parts.append("This dataset can be used to answer questions about:")
        context_parts.append("- Groundwater availability and extraction patterns across Indian states and districts")
        context_parts.append("- Rainfall patterns and their impact on groundwater recharge")
        context_parts.append("- Comparison of groundwater resources between different regions")
        context_parts.append("- Trends in groundwater extraction and sustainability")
        context_parts.append("- Water quality issues and contamination patterns")
        context_parts.append("- Future water resource planning and projections")
        context_parts.append("- Environmental flow requirements and conservation needs")

        # Column Descriptions (abbreviated)
        context_parts.append("\n## Key Column Descriptions")
        important_columns = [
            "STATE", "DISTRICT", "ASSESSMENT UNIT", "Rainfall (mm)_Total",
            "Annual Ground water Recharge (ham)_Total", "Annual Extractable Ground water Resource (ham)_Total",
            "Ground Water Extraction for all uses (ha.m)_Total", "Stage of Ground Water Extraction (%)_Total"
        ]

        for col in important_columns:
            if col in self.column_descriptions:
                context_parts.append(f"- **{col}**: {self.column_descriptions[col]}")

        return "\n".join(context_parts)

    def generate_context_file(self, output_file: str = "generalChatIndex.txt") -> str:
        """Generate the complete context file"""
        try:
            logger.info("Starting CSV analysis for general chat context...")
            analysis = self.analyze_csv_files()

            logger.info("Generating context summary...")
            context_summary = self.generate_context_summary(analysis)

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(context_summary)

            logger.info(f"Context file generated successfully: {output_file}")
            logger.info(f"Context length: {len(context_summary)} characters")

            return output_file

        except Exception as e:
            import traceback
            logger.error(f"Error generating context file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


class GeneralChatService:
    """Service for handling general chat queries using LLM and context"""

    def __init__(self, llm, context_file: str = "generalChatIndex.txt"):
        self.llm = llm
        self.context_file = context_file
        self.context = self._load_context()

    def _load_context(self) -> str:
        """Load context from file"""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"Context file {self.context_file} not found")
                return ""
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            return ""

    def refresh_context(self):
        """Refresh context from file"""
        self.context = self._load_context()

    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a general question using LLM and context"""
        try:
            if not self.llm:
                return {
                    "success": False,
                    "response": "LLM service not available",
                    "error": "LLM not configured"
                }

            if not self.context:
                return {
                    "success": False,
                    "response": "Context not available. Please ensure the dataset has been analyzed.",
                    "error": "Context not loaded"
                }

            # Create prompt with context
            prompt = f"""You are an AI assistant specialized in Indian groundwater resource data analysis. You have access to comprehensive groundwater assessment data for India.

Context Information:
{self.context}

User Question: {question}

Please provide a helpful, accurate response based on the context information above. If the question is outside the scope of the groundwater data, politely explain what information you can provide instead.

Response:"""

            # Get response from LLM
            response = await self.llm.ainvoke(prompt)

            return {
                "success": True,
                "response": response.content,
                "error": ""
            }

        except Exception as e:
            logger.error(f"Error processing general chat question: {e}")
            return {
                "success": False,
                "response": "Sorry, I encountered an error while processing your question.",
                "error": str(e)
            }


# Utility function to generate context if needed
def ensure_context_exists(csv_output_dir: str = "datasets/csv_output", context_file: str = "generalChatIndex.txt"):
    """Ensure context file exists, generate if needed"""
    if not os.path.exists(context_file):
        logger.info("Context file not found, generating...")
        generator = GeneralChatContextGenerator(csv_output_dir)
        generator.generate_context_file(context_file)
    return context_file