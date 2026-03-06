const API_BASE = window.location.origin;   
let totalPred  = 0;
let fraudCount = 0;
const history  = [];
 
function updateClock() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-GB', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();
 
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}
 
function copyApiUrl() {
  const url = document.getElementById('apiUrl').textContent;
  navigator.clipboard.writeText(url);
  showToast('✅ API URL copied to clipboard');
}
 
function fillLegit() {
  document.getElementById('txAmt').value       = '49.99';
  document.getElementById('productCD').value   = 'W';
  document.getElementById('card1').value       = '9500';
  document.getElementById('card4').value       = 'visa';
  document.getElementById('card6').value       = 'debit';
  document.getElementById('emailDomain').value = 'gmail.com';
  document.getElementById('c1').value          = '1';
  document.getElementById('c13').value         = '1';
  document.getElementById('c14').value         = '1';
  showToast('📋 Legit transaction loaded');
}

function fillSuspicious() {
  document.getElementById('txAmt').value       = '9999.99';
  document.getElementById('productCD').value   = 'C';
  document.getElementById('card1').value       = '4774';      // ← change this
  document.getElementById('card4').value       = 'american express';
  document.getElementById('card6').value       = 'credit';
  document.getElementById('emailDomain').value = 'protonmail.com';
  document.getElementById('c1').value          = '9';
  document.getElementById('c13').value         = '10';
  document.getElementById('c14').value         = '8';
  showToast('⚠️ Suspicious transaction loaded');
}
 
function getRiskClass(prob) {
  if (prob < 0.2)  return 'low';
  if (prob < 0.5)  return 'medium';
  if (prob < 0.75) return 'high';
  return 'critical';
}

function getRiskLabel(prob) {
  if (prob < 0.2)  return '🟢 LOW RISK';
  if (prob < 0.5)  return '🟡 MEDIUM RISK';
  if (prob < 0.75) return '🟠 HIGH RISK';
  return '🔴 CRITICAL RISK';
}
 
function renderResult(data) {
  const prob      = data.fraud_probability;
  const riskClass = getRiskClass(prob);
  const pct       = (prob * 100).toFixed(1) + '%';

  document.getElementById('resultEmpty').style.display = 'none';
  const rc = document.getElementById('resultContent');
  rc.classList.add('show');
 
  const badge = document.getElementById('riskBadge');
  badge.className   = `risk-badge risk-${riskClass}`;
  badge.textContent = getRiskLabel(prob);
 
  setTimeout(() => {
    const bar     = document.getElementById('probBar');
    bar.style.width = pct;
    bar.className   = `prob-bar-fill ${riskClass}`;
  }, 50);
  document.getElementById('probPct').textContent = pct;
 
  document.getElementById('txId').textContent       = data.transaction_id || '—';
  document.getElementById('confidence').textContent  = data.confidence || '—';
  document.getElementById('modelName').textContent   = (data.model_version || '').replace(' Tuned', '');
  document.getElementById('resultTime').textContent  =
    new Date(data.timestamp).toLocaleTimeString('en-GB', { hour12: false });
 
  document.getElementById('recommendation').textContent = data.recommendation || '—';
 
  renderShap(data.top_risk_factors || []);
 
  totalPred++;
  if (data.is_fraud) fraudCount++;
  document.getElementById('totalPred').textContent  = totalPred;
  document.getElementById('fraudCount').textContent = fraudCount;
  document.getElementById('fraudRate').textContent  =
    ((fraudCount / totalPred) * 100).toFixed(1) + '% rate';
 
  addHistory(data);
}
 
function renderShap(factors) {
  const list = document.getElementById('shapList');
  if (!factors.length) return;

  const maxShap = Math.max(...factors.map(f => Math.abs(f.shap_score || 0)));

  list.innerHTML = factors.map(f => {
    const pct = maxShap > 0 ? (Math.abs(f.shap_score) / maxShap * 100) : 0;
    const dir = (f.shap_score || 0) > 0 ? 'pos' : 'neg';
    const arrow = dir === 'pos' ? '↑' : '↓';
    return `
      <div class="shap-item">
        <div class="shap-name" title="${f.feature}">${f.feature}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${dir}" style="width:0%" data-width="${pct}%"></div>
        </div>
        <div class="shap-val">${arrow} ${Math.abs(f.shap_score || 0).toFixed(3)}</div>
      </div>`;
  }).join('');

  setTimeout(() => {
    list.querySelectorAll('.shap-bar-fill').forEach(el => {
      el.style.width = el.dataset.width;
    });
  }, 50);
}
 
function addHistory(data) {
  history.unshift(data);
  if (history.length > 10) history.pop();

  const tbody = document.getElementById('historyBody');
  tbody.innerHTML = history.map(d => {
    const prob = d.fraud_probability;
    const pct  = (prob * 100).toFixed(1) + '%';
    const rc   = getRiskClass(prob);
    const riskColors = {
      low: 'var(--accent3)', medium: 'var(--warn)',
      high: 'var(--accent2)', critical: 'var(--accent2)'
    };
    const time = new Date(d.timestamp).toLocaleTimeString('en-GB', { hour12: false });
    return `
      <tr>
        <td>${time}</td>
        <td>$${d.amount || '—'}</td>
        <td style="color:${riskColors[rc]};font-weight:bold">${pct}</td>
        <td style="color:${riskColors[rc]}">${rc.toUpperCase()}</td>
        <td>
          <span class="tag ${d.is_fraud ? 'tag-fraud' : 'tag-legit'}">
            ${d.is_fraud ? 'FRAUD' : 'LEGIT'}
          </span>
        </td>
      </tr>`;
  }).join('');
}
 
async function runPrediction() {
  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.add('show');

  const payload = {
    TransactionAmt : parseFloat(document.getElementById('txAmt').value)     || 100,
    ProductCD      : document.getElementById('productCD').value,
    card1          : parseInt(document.getElementById('card1').value)        || 9500,
    card4          : document.getElementById('card4').value,
    card6          : document.getElementById('card6').value,
    P_emaildomain  : document.getElementById('emailDomain').value,
    R_emaildomain  : document.getElementById('emailDomain').value,
    TransactionDT  : Math.floor(Date.now() / 1000) % (86400 * 180),
    C1             : parseFloat(document.getElementById('c1').value)  || 1,
    C13            : parseFloat(document.getElementById('c13').value) || 1,
    C14            : parseFloat(document.getElementById('c14').value) || 1,
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`API error: ${res.status}`);

    const data  = await res.json();
    data.amount = payload.TransactionAmt.toFixed(2);
    overlay.classList.remove('show');
    renderResult(data);
    showToast('Prediction complete');

  } catch (err) {
    overlay.classList.remove('show');
    showToast('API error: ' + err.message);
    console.error(err);
  }
}
 
document.getElementById('apiUrl').textContent = `${API_BASE}/predict`;