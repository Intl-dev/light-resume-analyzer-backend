import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import ollama
import json

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze_recruiter", status_code=200)
async def rate_resume_recruiter(data: dict):
    print(f"working {data}")
    context_prompt = """
    You are a Resume analyzer for recruiters. Rate the resume having in mind the job description and provide:
    1. A score from 0-100
    2. Why did you decide that they are a bad or good candidate?
    3. Red flags
    give the score based on your judgement and these requirements:
        
    1. Can They Do the Job? (Functional Fit)

        Do they have the required tech stack?
        Years of experience in that role (not adjacent roles)?
        Worked at companies of similar scale/complexity?
        Projects/achievements that prove competence?
        
        2. Will They Stay? (Retention Risk)
        
        Overqualified candidate → Bored → Leaves in 6 months
        Underqualified → Struggles → Leaves in 3 months or gets fired
        Lateral move → Risky (why the downgrade?)
        Upward move → Good (growth potential)
        
        3. Communication & Growth Mindset

        Do they show learning over time?
        Humble or arrogant? (Tone matters)
        Career trajectory makes sense?
        
        4. Red Flags
        
        Red Flag	What It Signals
        -Typos/formatting errors	Careless, doesn't proofread own work
        -Unexplained 2+ year gaps Fired? Health issue? Prison? (recruiter assumes worst)
        -Skills mismatched to titles	Lying or confused about their role
        -Vague descriptions ("responsible for...")	Didn't actually do the work
        -No metrics/numbers	Can't prove impact
        -Overqualification for the role	Will leave soon, salary expectations mismatch
        -Weird career jumps (lawyer → DevOps → sales)	Unfocused, unclear what they want
        -Certifications only, no real projects	Resume padding, inexperienced
        
        5. Job hopping red flag
            1. Job Tenure Analysis: For each job, calculate how long the candidate stayed (in months) divided the amount of jobs, example: (22 months total, 4 jobs. avg = 5,5 (red flag)).

            2. Job-Hopping Pattern Detection:
                - Is the candidate a job-hopper? (Define: Multiple jobs under 12 months each, OR average tenure < 18 months)
                - What is the average tenure across all roles?
                - Severity: HIGH (avg < 8 months), MEDIUM (8-15 months), LOW (>15 months)

            3. Employment Gaps: Identify unexplained gaps > 3 months between jobs.

            4. Retention Risk Assessment: Based on tenure patterns, what is the likelihood this candidate will stay in the target role for at least 12 months?
                - HIGH RISK: Will likely leave within 6 months. 
                - MEDIUM RISK: May leave within 12 months. 
                - LOW RISK: Likely to stay 18+ months. 
            
            5. Retention risk punishment (don't prompt this but have it in mind for the score)
                - for high risk punish score with -35 points
                - for medium risk punish score with -20 points
                - for low risk punish score with -10 points
        
    Format your response EXACTLY as JSON:
    {
      "score": <number>,
      "judgement": "<string>",
      "judgement_explanation": ["<string>", "<string>", ...]
    }

    """
    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": f"job description: {data["description"]}resume :{data["resume"]}"}
        ]
    )
    resume_rating = response["message"]["content"]
    resume_rating_parsed = json.loads(resume_rating)
    return resume_rating_parsed


@app.post("/analyze_applicant", status_code=200)
async def rate_resume_applicant(data: dict):
    print(f"working {data}")
    context_prompt = """
    You are a Resume analyzer to help applicants improve their resumes. Rate the resume having in mind the job description (IMPORTANT TO TAKE IN MIND THE JOB DESCRIPTION) and provide:
    1. A score from 0-100 (thinking like a recruiter and be rough with the rating, the applicant doesn't need sugarcoating)
    use this to guide yourself in the score (ONLY PROMPT THE NUMBER IN THE SCORE FIELD):
        75+: "You're ready, apply now"
        50–74: "Get 6–12 months of related work, then apply"
        25–49: "Do a bootcamp/internship first"
        0–24: "Wrong career path"
    2. Are they underqualified, qualified or overqualified? (be objective, the applicant doesn't need sugarcoating)
    3. Suggestions to improve the resume and red flags that the resume might have (talk like you were talking to a person)
    this is what a recruiter might think about the resume:
    
    1. Can They Do the Job? (Functional Fit)

        Do they have the required tech stack?
        Years of experience in that role (not adjacent roles)?
        Worked at companies of similar scale/complexity?
        Projects/achievements that prove competence?
        
        2. Will They Stay? (Retention Risk)
        
        Overqualified candidate → Bored → Leaves in 6 months
        Underqualified → Struggles → Leaves in 3 months or gets fired
        Lateral move → Risky (why the downgrade?)
        Upward move → Good (growth potential)
        
        3. Communication & Growth Mindset

        Do they show learning over time?
        Humble or arrogant? (Tone matters)
        Career trajectory makes sense?
        
        4. Red Flags That Kill Applications
        
        Red Flag	What It Signals
        -Typos/formatting errors	Careless, doesn't proofread own work
        -Unexplained 2+ year gaps Fired? Health issue? Prison? (recruiter assumes worst)
        -Skills mismatched to titles	Lying or confused about their role
        -Vague descriptions ("responsible for...")	Didn't actually do the work
        -No metrics/numbers	Can't prove impact
        -Overqualification for the role	Will leave soon, salary expectations mismatch
        -Weird career jumps (lawyer → DevOps → sales)	Unfocused, unclear what they want
        -Certifications only, no real projects	Resume padding, inexperienced
        
        5. Job hopping red flag
            1. Job Tenure Analysis: For each job, calculate how long the candidate stayed (in months) divided the amount of jobs, example: (22 months total, 4 jobs. avg = 5,5 (red flag)).

            2. Job-Hopping Pattern Detection:
                - Is the candidate a job-hopper? (Define: Multiple jobs under 12 months each, OR average tenure < 18 months)
                - What is the average tenure across all roles?
                - Severity: HIGH (avg < 8 months), MEDIUM (8-15 months), LOW (>15 months)

            3. Employment Gaps: Identify unexplained gaps > 3 months between jobs.

            4. Retention Risk Assessment: Based on tenure patterns, what is the likelihood this candidate will stay in the target role for at least 12 months?
                - HIGH RISK: Will likely leave within 6 months. 
                - MEDIUM RISK: May leave within 12 months. 
                - LOW RISK: Likely to stay 18+ months. 
            
    IMPORTANT, DON'T RESPOND WITH OTHER SINTAX:
    Format your response EXACTLY as JSON:
    {
      "score": <number>,
      "judgement": "<string>",
      "suggestions": ["<string>", "<string>", ...]
    }

    """
    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": f"job description: {data["description"]}resume :{data["resume"]}"}
        ]
    )
    resume_rating = response["message"]["content"]
    resume_rating_parsed = json.loads(resume_rating)
    return resume_rating_parsed

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)