from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List

from ...database import get_db
from ...models import Composite, ECMAttempt, Factor

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """
    Simple dashboard showing all factorization results.
    """
    
    # Get recent composites with their attempts and factors
    composites = db.query(Composite).order_by(desc(Composite.created_at)).limit(50).all()
    
    # Get recent attempts
    attempts = db.query(ECMAttempt).order_by(desc(ECMAttempt.created_at)).limit(100).all()
    
    # Get all factors
    factors = db.query(Factor).order_by(desc(Factor.created_at)).all()
    
    # Build summary stats
    total_composites = db.query(Composite).count()
    total_attempts = db.query(ECMAttempt).count()
    total_factors = db.query(Factor).count()
    fully_factored = db.query(Composite).filter(Composite.is_fully_factored == True).count()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECM Distributed Factorization Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .stats {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .stat-item {{ display: inline-block; margin-right: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .number {{ font-family: monospace; word-break: break-all; }}
            .factor {{ color: #d63384; font-weight: bold; }}
            .success {{ color: #198754; }}
            .pending {{ color: #fd7e14; }}
            .section {{ margin-bottom: 40px; }}
            .small-text {{ font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <h1>ECM Distributed Factorization Dashboard</h1>
        
        <div class="stats">
            <h3>Summary Statistics</h3>
            <div class="stat-item"><strong>Total Composites:</strong> {total_composites}</div>
            <div class="stat-item"><strong>Total Attempts:</strong> {total_attempts}</div>
            <div class="stat-item"><strong>Factors Found:</strong> {total_factors}</div>
            <div class="stat-item"><strong>Fully Factored:</strong> {fully_factored}</div>
        </div>

        <div class="section">
            <h2>Recent Composites</h2>
            <table>
                <thead>
                    <tr>
                        <th>Number</th>
                        <th>Bit Length</th>
                        <th>Status</th>
                        <th>Attempts</th>
                        <th>Factors</th>
                        <th>Added</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for comp in composites:
        attempt_count = db.query(ECMAttempt).filter(ECMAttempt.composite_id == comp.id).count()
        comp_factors = db.query(Factor).filter(Factor.composite_id == comp.id).all()
        
        if comp.is_prime:
            status = '<span class="success">Prime</span>'
        elif comp.is_fully_factored:
            status = '<span class="success">Fully Factored</span>'
        else:
            status = '<span class="pending">Composite</span>'
        
        factors_display = ""
        if comp_factors:
            factors_display = " × ".join([f'<span class="factor">{f.factor}</span>' for f in comp_factors])
        
        html_content += f"""
                    <tr>
                        <td class="number">{comp.number[:50]}{'...' if len(comp.number) > 50 else ''}</td>
                        <td>{comp.digit_length}</td>
                        <td>{status}</td>
                        <td>{attempt_count}</td>
                        <td>{factors_display or 'None'}</td>
                        <td class="small-text">{comp.created_at.strftime('%Y-%m-%d %H:%M')}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Recent Factorization Attempts</h2>
            <table>
                <thead>
                    <tr>
                        <th>Composite</th>
                        <th>Method</th>
                        <th>B1</th>
                        <th>Curves</th>
                        <th>Factor Found</th>
                        <th>Client</th>
                        <th>Time</th>
                        <th>Submitted</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for attempt in attempts:
        comp = db.query(Composite).filter(Composite.id == attempt.composite_id).first()
        comp_display = comp.number[:30] + '...' if comp and len(comp.number) > 30 else (comp.number if comp else 'Unknown')
        
        factor_display = "None"
        if attempt.factor_found:
            factor_display = f'<span class="factor">{attempt.factor_found}</span>'
        
        html_content += f"""
                    <tr>
                        <td class="number">{comp_display}</td>
                        <td>{attempt.method}</td>
                        <td>{attempt.b1 or 'N/A'}</td>
                        <td>{attempt.curves_completed}</td>
                        <td>{factor_display}</td>
                        <td class="small-text">{attempt.client_id or 'Unknown'}</td>
                        <td>{attempt.execution_time_seconds:.1f}s</td>
                        <td class="small-text">{attempt.created_at.strftime('%Y-%m-%d %H:%M')}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>All Factors Found</h2>
            <table>
                <thead>
                    <tr>
                        <th>Factor</th>
                        <th>Composite</th>
                        <th>Discovery Method</th>
                        <th>Discovered</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for factor in factors:
        comp = db.query(Composite).filter(Composite.id == factor.composite_id).first()
        attempt = db.query(ECMAttempt).filter(ECMAttempt.id == factor.found_by_attempt_id).first()
        
        comp_display = comp.number[:40] + '...' if comp and len(comp.number) > 40 else (comp.number if comp else 'Unknown')
        method = attempt.method if attempt else 'Unknown'
        
        html_content += f"""
                    <tr>
                        <td class="factor number">{factor.factor}</td>
                        <td class="number">{comp_display}</td>
                        <td>{method}</td>
                        <td class="small-text">{factor.created_at.strftime('%Y-%m-%d %H:%M')}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>

        <div class="small-text">
            <p>Dashboard auto-refreshes every 30 seconds. API Documentation: <a href="/docs">/docs</a></p>
        </div>

        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function() {
                window.location.reload();
            }, 30000);
        </script>
    </body>
    </html>
    """
    
    return html_content