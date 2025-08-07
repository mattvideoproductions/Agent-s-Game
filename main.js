/*
  Bathtub Fluid with Ducks
  - Semi-Lagrangian advection + diffusion + pressure solve (Jacobi)
  - Scalar dye for water amount; velocity field drives ducks
*/

const canvas = document.getElementById('sim');
const ctx = canvas.getContext('2d', { alpha: false });

// Simulation parameters
const SIM = {
  gridCols: 160, // resolution across width
  gridRows: 100, // resolution across height
  dt: 1 / 60,
  viscosity: 0.001,
  diffusion: 0.0002,
  buoyancy: 18,
  gravity: 70,
  forceScale: 140,
  pressureIters: 30,
};

const ui = {
  force: document.getElementById('rangeForce'),
  visc: document.getElementById('rangeVisc'),
  buoy: document.getElementById('rangeBuoy'),
  rain: document.getElementById('chkRain'),
  addDuck: document.getElementById('btnAddDuck'),
  reset: document.getElementById('btnReset'),
};

const WIDTH = canvas.width;
const HEIGHT = canvas.height;
const cellW = WIDTH / SIM.gridCols;
const cellH = HEIGHT / SIM.gridRows;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Grid helpers
function idx(x, y) { return x + y * SIM.gridCols; }
function forEachCell(fn) {
  for (let y = 0; y < SIM.gridRows; y++) {
    for (let x = 0; x < SIM.gridCols; x++) {
      fn(x, y);
    }
  }
}

// Fields
let velX = new Float32Array(SIM.gridCols * SIM.gridRows);
let velY = new Float32Array(SIM.gridCols * SIM.gridRows);
let velX0 = new Float32Array(SIM.gridCols * SIM.gridRows);
let velY0 = new Float32Array(SIM.gridCols * SIM.gridRows);
let dye = new Float32Array(SIM.gridCols * SIM.gridRows);
let dye0 = new Float32Array(SIM.gridCols * SIM.gridRows);
let pressure = new Float32Array(SIM.gridCols * SIM.gridRows);
let divergence = new Float32Array(SIM.gridCols * SIM.gridRows);

function clearFields() {
  velX.fill(0); velY.fill(0);
  velX0.fill(0); velY0.fill(0);
  dye.fill(0); dye0.fill(0);
  pressure.fill(0); divergence.fill(0);
}

// Boundary conditions: solid walls with no-slip
function setBoundary(vx, vy) {
  // Left/Right walls
  for (let y = 0; y < SIM.gridRows; y++) {
    vx[idx(0, y)] = 0; vx[idx(SIM.gridCols - 1, y)] = 0;
    vy[idx(0, y)] = -vy[idx(1, y)];
    vy[idx(SIM.gridCols - 1, y)] = -vy[idx(SIM.gridCols - 2, y)];
  }
  // Top/Bottom walls
  for (let x = 0; x < SIM.gridCols; x++) {
    vy[idx(x, 0)] = 0; vy[idx(x, SIM.gridRows - 1)] = 0;
    vx[idx(x, 0)] = -vx[idx(x, 1)];
    vx[idx(x, SIM.gridRows - 1)] = -vx[idx(x, SIM.gridRows - 2)];
  }
}

function diffuse(b, x, x0, diff, dt) {
  const a = dt * diff * SIM.gridCols * SIM.gridRows;
  // Jacobi iterations
  for (let k = 0; k < 15; k++) {
    for (let y = 1; y < SIM.gridRows - 1; y++) {
      for (let xC = 1; xC < SIM.gridCols - 1; xC++) {
        const i = idx(xC, y);
        x[i] = (x0[i] + a * (x[i - 1] + x[i + 1] + x[i - SIM.gridCols] + x[i + SIM.gridCols])) / (1 + 4 * a);
      }
    }
  }
  if (b) setBoundary(velX, velY);
}

function advect(b, d, d0, vx, vy, dt) {
  for (let y = 1; y < SIM.gridRows - 1; y++) {
    for (let xC = 1; xC < SIM.gridCols - 1; xC++) {
      const i = idx(xC, y);
      let x = xC - dt * vx[i] / cellW;
      let yb = y - dt * vy[i] / cellH;
      x = clamp(x, 0.5, SIM.gridCols - 1.5);
      yb = clamp(yb, 0.5, SIM.gridRows - 1.5);
      const x0f = Math.floor(x);
      const y0f = Math.floor(yb);
      const x1f = x0f + 1;
      const y1f = y0f + 1;
      const sx = x - x0f; const sy = yb - y0f;
      const i00 = idx(x0f, y0f);
      const i10 = idx(x1f, y0f);
      const i01 = idx(x0f, y1f);
      const i11 = idx(x1f, y1f);
      d[i] = (1 - sx) * (1 - sy) * d0[i00] + sx * (1 - sy) * d0[i10] + (1 - sx) * sy * d0[i01] + sx * sy * d0[i11];
    }
  }
  if (b) setBoundary(velX, velY);
}

function computeDivergence() {
  for (let y = 1; y < SIM.gridRows - 1; y++) {
    for (let xC = 1; xC < SIM.gridCols - 1; xC++) {
      const i = idx(xC, y);
      const vxr = velX[idx(xC + 1, y)] - velX[idx(xC - 1, y)];
      const vyb = velY[idx(xC, y + 1)] - velY[idx(xC, y - 1)];
      divergence[i] = (vxr / (2 * cellW)) + (vyb / (2 * cellH));
    }
  }
}

function pressureSolve() {
  pressure.fill(0);
  for (let k = 0; k < SIM.pressureIters; k++) {
    for (let y = 1; y < SIM.gridRows - 1; y++) {
      for (let xC = 1; xC < SIM.gridCols - 1; xC++) {
        const i = idx(xC, y);
        pressure[i] = (divergence[i] + pressure[i - 1] + pressure[i + 1] + pressure[i - SIM.gridCols] + pressure[i + SIM.gridCols]) / 4;
      }
    }
  }
}

function subtractPressureGradient() {
  for (let y = 1; y < SIM.gridRows - 1; y++) {
    for (let xC = 1; xC < SIM.gridCols - 1; xC++) {
      const i = idx(xC, y);
      const gradX = (pressure[idx(xC + 1, y)] - pressure[idx(xC - 1, y)]) / (2 * cellW);
      const gradY = (pressure[idx(xC, y + 1)] - pressure[idx(xC, y - 1)]) / (2 * cellH);
      velX[i] -= gradX;
      velY[i] -= gradY;
    }
  }
  setBoundary(velX, velY);
}

function addForces() {
  // Viscosity (diffusion on velocity)
  velX0.set(velX); velY0.set(velY);
  diffuse(true, velX, velX0, SIM.viscosity, SIM.dt);
  diffuse(true, velY, velY0, SIM.viscosity, SIM.dt);

  // Gravity and buoyancy from dye amount
  forEachCell((xC, y) => {
    const i = idx(xC, y);
    const liquid = dye[i];
    // Heavier water pulls down; buoyant region pushes up
    velY[i] += SIM.dt * (SIM.gravity * 0.2 + (-SIM.buoyancy * liquid));
  });

  // Simple wall tap to keep water in bounds (no-slip already handles)
  setBoundary(velX, velY);
}

function project() {
  computeDivergence();
  pressureSolve();
  subtractPressureGradient();
}

function step() {
  // External forces
  addForces();

  // Advect velocity
  velX0.set(velX); velY0.set(velY);
  advect(true, velX, velX0, velX0, velY0, SIM.dt);
  advect(true, velY, velY0, velX0, velY0, SIM.dt);
  setBoundary(velX, velY);

  // Make velocity divergence-free
  project();

  // Advect dye
  dye0.set(dye);
  advect(false, dye, dye0, velX, velY, SIM.dt);

  // Diffuse dye a touch
  diffuse(false, dye, dye0, SIM.diffusion, SIM.dt);

  // Mild decay to avoid infinite fill
  for (let i = 0; i < dye.length; i++) dye[i] *= 0.9995;
}

// Interaction
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
canvas.addEventListener('mousedown', (e) => { isDragging = true; lastMouse = getMouse(e); });
window.addEventListener('mouseup', () => isDragging = false);
canvas.addEventListener('mousemove', (e) => {
  const pos = getMouse(e);
  if (isDragging) {
    const dx = pos.x - lastMouse.x;
    const dy = pos.y - lastMouse.y;
    stirAndPour(pos.x, pos.y, dx, dy, e.shiftKey);
  }
  lastMouse = pos;
});

function getMouse(e) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function stirAndPour(px, py, dx, dy, draining) {
  const radius = 36;
  const force = SIM.forceScale;
  const sign = draining ? -1 : 1;
  forEachCell((xC, yC) => {
    const cx = (xC + 0.5) * cellW;
    const cy = (yC + 0.5) * cellH;
    const dist = Math.hypot(px - cx, py - cy);
    if (dist < radius) {
      const i = idx(xC, yC);
      const falloff = Math.max(0, 1 - dist / radius);
      const impulse = force * falloff * 0.0005;
      velX[i] += (dx / (SIM.dt + 1e-6)) * impulse;
      velY[i] += (dy / (SIM.dt + 1e-6)) * impulse;
      dye[i] = clamp(dye[i] + sign * 0.9 * falloff, 0, 1.5);
    }
  });
}

// Rain
let rainPhase = 0;
function maybeRain() {
  if (!ui.rain.checked) return;
  rainPhase += SIM.dt;
  const drops = 6;
  for (let d = 0; d < drops; d++) {
    const xPixel = (Math.sin(rainPhase * 1.3 + d) * 0.5 + 0.5) * WIDTH;
    const yPixel = 40 + (d % 3) * 10;
    stirAndPour(xPixel, yPixel, 0, 80, false);
  }
}

// Ducks
const ducks = [];
const DUCK = {
  radius: 14,
  density: 0.6, // relative to water; lower floats more
  drag: 0.9,
};

function spawnDuck(x, y) {
  ducks.push({
    x, y,
    vx: 0, vy: 0,
    spin: Math.random() * Math.PI * 2,
    hue: 40 + Math.random() * 15,
  });
}

ui.addDuck.addEventListener('click', () => spawnDuck(WIDTH * 0.5 + (Math.random() - 0.5) * 80, 40 + Math.random() * 20));
ui.reset.addEventListener('click', () => { clearFields(); ducks.length = 0; });

ui.force.addEventListener('input', () => SIM.forceScale = Number(ui.force.value));
ui.visc.addEventListener('input', () => SIM.viscosity = Number(ui.visc.value));
ui.buoy.addEventListener('input', () => SIM.buoyancy = Number(ui.buoy.value));

// A few default ducks
for (let i = 0; i < 4; i++) spawnDuck(WIDTH * (0.4 + 0.2 * i), 120 + i * 2);

function sampleVelocity(px, py) {
  const x = clamp(px / cellW - 0.5, 0, SIM.gridCols - 1);
  const y = clamp(py / cellH - 0.5, 0, SIM.gridRows - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, SIM.gridCols - 1);
  const y1 = Math.min(y0 + 1, SIM.gridRows - 1);
  const sx = x - x0, sy = y - y0;
  const i00 = idx(x0, y0), i10 = idx(x1, y0), i01 = idx(x0, y1), i11 = idx(x1, y1);
  const vx = (1 - sx) * (1 - sy) * velX[i00] + sx * (1 - sy) * velX[i10] + (1 - sx) * sy * velX[i01] + sx * sy * velX[i11];
  const vy = (1 - sx) * (1 - sy) * velY[i00] + sx * (1 - sy) * velY[i10] + (1 - sx) * sy * velY[i01] + sx * sy * velY[i11];
  return { vx, vy };
}

function sampleDye(px, py) {
  const x = clamp(px / cellW - 0.5, 0, SIM.gridCols - 1);
  const y = clamp(py / cellH - 0.5, 0, SIM.gridRows - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, SIM.gridCols - 1);
  const y1 = Math.min(y0 + 1, SIM.gridRows - 1);
  const sx = x - x0, sy = y - y0;
  const i00 = idx(x0, y0), i10 = idx(x1, y0), i01 = idx(x0, y1), i11 = idx(x1, y1);
  const v = (1 - sx) * (1 - sy) * dye[i00] + sx * (1 - sy) * dye[i10] + (1 - sx) * sy * dye[i01] + sx * sy * dye[i11];
  return v;
}

function stepDucks() {
  for (const d of ducks) {
    const { vx, vy } = sampleVelocity(d.x, d.y);
    const localDye = sampleDye(d.x, d.y);
    const buoy = (1 - DUCK.density) * SIM.buoyancy * (0.5 + 0.5 * clamp(localDye, 0, 1));
    const g = SIM.gravity * 0.8;

    // Apply forces
    d.vx += (vx - d.vx) * 0.5; // follow flow
    d.vy += (vy - d.vy) * 0.5;
    d.vy += (buoy - g) * SIM.dt;

    // Drag
    d.vx *= DUCK.drag; d.vy *= DUCK.drag;

    d.x += d.vx * SIM.dt * 20;
    d.y += d.vy * SIM.dt * 20;

    // Wall collisions
    if (d.x < DUCK.radius) { d.x = DUCK.radius; d.vx *= -0.3; }
    if (d.x > WIDTH - DUCK.radius) { d.x = WIDTH - DUCK.radius; d.vx *= -0.3; }
    if (d.y < DUCK.radius) { d.y = DUCK.radius; d.vy *= -0.3; }
    if (d.y > HEIGHT - DUCK.radius) { d.y = HEIGHT - DUCK.radius; d.vy *= -0.5; }

    d.spin += 0.5 * ((Math.random() - 0.5) + 0.001 * d.vx);
  }
}

// Rendering
const imgData = ctx.createImageData(WIDTH, HEIGHT);
const pixels = imgData.data;

function renderDye() {
  for (let y = 0; y < HEIGHT; y++) {
    const gy = Math.floor(y / cellH);
    for (let x = 0; x < WIDTH; x++) {
      const gx = Math.floor(x / cellW);
      const iGrid = idx(clamp(gx, 0, SIM.gridCols - 1), clamp(gy, 0, SIM.gridRows - 1));
      const v = dye[iGrid];
      const c = Math.max(0, Math.min(1, v));
      // sky-blue water with foam
      const r = 180 - c * 40;
      const g = 210 + c * 20;
      const b = 255 - c * 30;
      const j = (x + y * WIDTH) * 4;
      pixels[j] = r;
      pixels[j + 1] = g;
      pixels[j + 2] = b;
      pixels[j + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

function drawDuck(d) {
  const r = DUCK.radius;
  ctx.save();
  ctx.translate(d.x, d.y);
  ctx.rotate(Math.sin(d.spin) * 0.15);
  // body
  ctx.fillStyle = `hsl(${d.hue}deg 90% 55%)`;
  ctx.beginPath();
  ctx.ellipse(0, 0, r, r * 0.75, 0, 0, Math.PI * 2);
  ctx.fill();
  // head
  ctx.beginPath();
  ctx.arc(r * 0.7, -r * 0.4, r * 0.45, 0, Math.PI * 2);
  ctx.fill();
  // beak
  ctx.fillStyle = `hsl(25deg 90% 45%)`;
  ctx.beginPath();
  ctx.ellipse(r * 1.25, -r * 0.35, r * 0.35, r * 0.2, 0, 0, Math.PI * 2);
  ctx.fill();
  // eye
  ctx.fillStyle = '#111';
  ctx.beginPath();
  ctx.arc(r * 0.92, -r * 0.5, r * 0.08, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function renderDucks() {
  for (const d of ducks) drawDuck(d);
}

function frame() {
  SIM.dt = 1 / 60; // fixed for stability
  maybeRain();
  step();
  stepDucks();
  renderDye();
  renderDucks();
  requestAnimationFrame(frame);
}

clearFields();
frame();