# Software Project Effort and Cost Estimation - Dataset Requirements

## Overview
This document outlines the required features and their specifications for software project effort and cost estimation using our hybrid estimation model. Following these guidelines ensures accurate predictions and optimal model performance.

## Required Features

### Core Project Metrics

| Feature Name | Type | Unit | Range | Description | Required |
|-------------|------|------|--------|-------------|----------|
| Object_points | Integer | Points | 20-1000 | Measure of project size based on UI elements, data entities, and system interactions | Yes |
| Team_size | Integer | People | 2-20 | Total number of team members assigned to the project | Yes |
| Estimated_duration | Integer | Months | 1-24 | Initial estimated project duration | Yes |
| Actual_duration | Integer | Months | 1-24 | Actual project duration | Yes |
| Daily_hours | Float | Hours | 4-12 | Average working hours per day | Yes |
| Development_type | String | - | ['New', 'Enhancement', 'Maintenance'] | Type of development project | Yes |
| Dedicated_team_members | Integer | People | 1-20 | Number of full-time team members | Yes |
| Team_experience | String | - | ['Low', 'Medium', 'High'] | Overall team experience level | Yes |
| Project_complexity | String | - | ['Low', 'Medium', 'High'] | Project complexity level | Yes |
| Domain_category | String | - | ['Business', 'Engineering', 'System'] | Project domain type | Yes |
| Estimated_size | Float | Points | 10-800 | Initial size estimate (if Object_points not available) | No |
| Development_platform | String | - | ['Web', 'Desktop', 'Mobile', 'Embedded'] | Target platform | Yes |

### Feature Specifications

#### Object Points
- **Definition**: Measure of project size based on:
  - Number of user interfaces
  - Number of data entities
  - System interactions
  - Business logic complexity
- **Calculation**: Sum of weighted components
- **Example Range**:
  - Small Project: 20-100
  - Medium Project: 100-300
  - Large Project: 300-1000

#### Team Composition
- **Team_size**: Total team members (including part-time)
- **Dedicated_team_members**: Must be ≤ Team_size
- **Ratio Guideline**: Dedicated members should be ≥ 50% of team size

#### Experience Levels
- **Low**: < 2 years relevant experience
- **Medium**: 2-5 years relevant experience
- **High**: > 5 years relevant experience

#### Project Complexity
- **Low**:
  - Simple CRUD operations
  - Basic user interfaces
  - Minimal integrations
- **Medium**:
  - Multiple integrations
  - Complex business logic
  - Data processing requirements
- **High**:
  - Real-time processing
  - Complex algorithms
  - Multiple external system dependencies
  - High security requirements

## Data Format Requirements

### File Format
- Preferred: Excel (.xlsx)
- Alternate: CSV (.csv)

### Data Quality Guidelines
1. **No Missing Values**: All required fields must be populated
2. **Data Types**: Must match specified types
3. **Range Validation**: Values must fall within specified ranges
4. **Consistency**: Units must be consistent across all entries

## Example Dataset Entry
```json
{
    "Object_points": 150,
    "Team_size": 6,
    "Duration": 8,
    "Daily_hours": 8,
    "Development_type": "New",
    "Dedicated_team_members": 4,
    "Team_experience": "Medium",
    "Project_complexity": "Medium"
}
```

## Model Output

### Effort Estimation
The model will provide three types of estimates:
1. **COCOMO Estimate**: Based on object points and complexity
2. **Dynamic Estimate**: Based on team composition and productivity
3. **Hybrid Estimate**: Weighted combination with adjustments

### Output Format
```json
{
    "COCOMO_Estimate": "hours",
    "Dynamic_Estimate": "hours",
    "Hybrid_Estimate": "hours",
    "Complexity_Score": "float (0.5-2.0)",
    "Team_Productivity": "hours per month"
}
```

## API Integration Guidelines

### Request Format
```http
POST /api/estimate
Content-Type: application/json

{
    // Include all required features as specified above
}
```

### Response Format
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "estimates": {
        // Estimation results as specified in Output Format
    },
    "metadata": {
        "complexity_score": float,
        "team_productivity": float
    }
}
```

## Data Validation Rules

1. **Object Points Validation**
   - Must be positive integer
   - Should correlate with project complexity

2. **Team Composition Validation**
   - Dedicated members ≤ Total team size
   - Minimum 2 team members total

3. **Duration Validation**
   - Must be positive integer
   - Maximum 24 months recommended

4. **Working Hours Validation**
   - Between 4-12 hours per day
   - Should consider local labor laws

## Best Practices

1. **Data Collection**
   - Use standardized forms/templates
   - Validate data at entry point
   - Document any assumptions

2. **Regular Updates**
   - Update team composition changes
   - Track actual vs estimated metrics
   - Document deviations

3. **Quality Assurance**
   - Regular data audits
   - Validation of input ranges
   - Cross-verification of related fields

## Notes
- All time-based calculations assume a standard 22 working days per month
- Cost calculations can be derived by multiplying effort hours with hourly rates
- Team productivity factors in both quantity and quality aspects
- Complexity scores are normalized between 0.5 and 2.0

For additional support or questions, please contact the development team. 