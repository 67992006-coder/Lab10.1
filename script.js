/**
 * Sign Pose Trainer – เครื่องมือฝึกจำท่ามือเดี่ยว (Static Single-Hand Pose)
 * ประมวลผลทั้งหมดในเบราว์เซอร์ · ไม่ส่งข้อมูลออกนอกเครื่อง
 *
 * ใช้:
 * - ml5.js handpose → จับ 21 จุดข้อต่อมือ
 * - ml5.js KNNClassifier → ให้ผู้ใช้สอนท่าและทำนาย
 * - IndexedDB → เก็บท่าที่สอนไว้ถาวรข้ามเซสชัน
 */

// ==================== ค่าคงที่ ====================
const CONFIDENCE_THRESHOLD = 0.7;
const DB_NAME = "SignPoseTrainerDB";
const DB_VERSION = 1;
const STORE_NAME = "signs";

// ==================== ตัวแปรสถานะ ====================
let video;
let canvas;
let ctx;
let handposeModel = null;
let knnClassifier = null;
let isPredictMode = true;
let currentHandFeatures = null;
let hasHand = false;
let frameCount = 0;
let skipFrames = false;
let isModelReady = false;
let db = null;
let latestPredictions = []; // เก็บผลล่าสุดไว้วาด

let signsData = {};
let exampleCounts = {};

// ==================== เริ่มต้น ====================
async function init() {
  video = document.getElementById("video");
  canvas = document.getElementById("canvas");
  ctx = canvas.getContext("2d");

  bindUIEvents();
  await openDatabase();
  await loadSignsFromDB();
  await startCamera();

  // โหลดโมเดลหลังกล้องพร้อม
  loadModels();

  // เริ่ม loop วาดภาพ
  requestAnimationFrame(drawLoop);
}

// ==================== เปิดกล้อง ====================
async function startCamera() {
  const statusEl = document.getElementById("camera-status");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        // ตั้งขนาด canvas ให้ตรงกับวิดีโอ
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });

    // บังคับเล่นวิดีโอ (บางเบราว์เซอร์ต้องการ)
    await video.play();

    statusEl.textContent = "กล้องพร้อม · กำลังโหลดโมเดล...";
    statusEl.classList.add("ok");
    statusEl.classList.remove("error");

    stream.getVideoTracks()[0].onended = () => {
      statusEl.textContent = "กล้องถูกปิดหรือหลุดการเชื่อมต่อ – กรุณารีเฟรชหน้า";
      statusEl.classList.add("error");
      statusEl.classList.remove("ok");
    };
  } catch (err) {
    console.error("Camera error:", err);
    let msg = "ไม่สามารถเปิดกล้องได้";

    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      msg = "คุณปฏิเสธสิทธิ์กล้อง – กรุณาอนุญาตแล้วรีเฟรชหน้า";
    } else if (err.name === "NotFoundError") {
      msg = "ไม่พบกล้องบนอุปกรณ์นี้";
    } else if (err.name === "NotReadableError") {
      msg = "กล้องถูกใช้งานโดยโปรแกรมอื่นอยู่";
    }

    statusEl.textContent = msg;
    statusEl.classList.add("error");
    statusEl.classList.remove("ok");
  }
}

// ==================== โหลดโมเดล ml5 ====================
function loadModels() {
  const statusEl = document.getElementById("camera-status");

  if (typeof ml5 === "undefined") {
    statusEl.textContent = "โหลด ml5.js ไม่สำเร็จ – ตรวจสอบเน็ตแล้วรีเฟรชหน้า";
    statusEl.classList.add("error");
    console.error("ml5 is not defined");
    return;
  }

  statusEl.textContent = "กำลังโหลดโมเดล AI... (รอ 15–40 วินาที)";
  console.log("เริ่มโหลด handpose...");

  try {
    knnClassifier = ml5.KNNClassifier();

    // ใช้ callback style ที่เสถียรกว่า
    handposeModel = ml5.handpose(video, { flipHorizontal: true }, modelReady);

  } catch (err) {
    console.error("สร้าง handpose ไม่สำเร็จ:", err);
    statusEl.textContent = "สร้างโมเดลไม่สำเร็จ: " + err.message;
    statusEl.classList.add("error");
  }
}

/**
 * เรียกเมื่อโมเดลโหลดเสร็จ
 */
function modelReady() {
  console.log("Handpose model พร้อมแล้ว!");
  const statusEl = document.getElementById("camera-status");
  statusEl.textContent = "โมเดลพร้อม · ยกมือขึ้นหน้ากล้อง";
  statusEl.classList.add("ok");
  statusEl.classList.remove("error");
  isModelReady = true;

  // เริ่มฟังผลทำนาย
  handposeModel.on("predict", (results) => {
    latestPredictions = results;
    handlePredictions(results);
  });
}

/**
 * จัดการผลจาก handpose
 */
function handlePredictions(predictions) {
  frameCount++;
  if (skipFrames && frameCount % 2 !== 0) return;

  if (predictions && predictions.length > 0) {
    hasHand = true;
    const landmarks = predictions[0].landmarks;
    currentHandFeatures = normalizeLandmarks(landmarks);

    if (isPredictMode && Object.keys(signsData).length > 0) {
      classifyHand(currentHandFeatures);
    }
  } else {
    hasHand = false;
    currentHandFeatures = null;
    if (isPredictMode) {
      document.getElementById("prediction-display").textContent = "–";
      document.getElementById("confidence-display").textContent = "–";
    }
  }

  updateHandStatus(hasHand);
}

/**
 * Normalize landmarks
 * - ย้าย origin ไปที่ข้อมือ
 * - scale ด้วยระยะข้อมือ → ปลายนิ้วกลาง
 */
function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const wrist = landmarks[0];
  const middleTip = landmarks[12];

  let scale = Math.hypot(
    middleTip[0] - wrist[0],
    middleTip[1] - wrist[1],
    middleTip[2] - wrist[2]
  );
  if (scale < 1e-6) scale = 1;

  const normalized = [];
  for (let i = 0; i < landmarks.length; i++) {
    const [x, y, z] = landmarks[i];
    normalized.push(
      (x - wrist[0]) / scale,
      (y - wrist[1]) / scale,
      (z - wrist[2]) / scale
    );
  }
  return normalized;
}

function classifyHand(features) {
  if (!features || !knnClassifier) return;

  knnClassifier.classify(features, (err, result) => {
    if (err) {
      console.error("Classify error:", err);
      return;
    }

    const label = result.label;
    const confidence = result.confidencesByLabel[label] || 0;

    const predEl = document.getElementById("prediction-display");
    const confEl = document.getElementById("confidence-display");

    if (confidence >= CONFIDENCE_THRESHOLD) {
      predEl.textContent = label;
      confEl.textContent = (confidence * 100).toFixed(1) + "%";
    } else {
      predEl.textContent = "ไม่แน่ใจ";
      confEl.textContent = (confidence * 100).toFixed(1) + "%";
    }
  });
}

// ==================== วาด Landmarks ====================
function drawLandmarks(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!predictions || predictions.length === 0) return;

  const landmarks = predictions[0].landmarks;

  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17]
  ];

  // วาดเส้น
  ctx.strokeStyle = "rgba(59, 130, 246, 0.9)";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

  for (const [a, b] of connections) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }

  // วาดจุด
  for (let i = 0; i < landmarks.length; i++) {
    const [x, y] = landmarks[i];
    ctx.beginPath();
    ctx.arc(x, y, i === 0 ? 7 : 5, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#22c55e" : "#60a5fa";
    ctx.fill();
  }
}

/**
 * Loop หลัก – วาด landmarks ทุกเฟรม
 */
function drawLoop() {
  // วาดจากผลล่าสุดที่มี
  if (isModelReady) {
    drawLandmarks(latestPredictions);
  }
  requestAnimationFrame(drawLoop);
}

// ==================== UI Events ====================
function bindUIEvents() {
  document.getElementById("btn-predict-mode").addEventListener("click", () => setMode(true));
  document.getElementById("btn-teach-mode").addEventListener("click", () => setMode(false));
  document.getElementById("btn-add-example").addEventListener("click", addExample);

  document.getElementById("sign-label").addEventListener("input", (e) => {
    const hasText = e.target.value.trim().length > 0;
    document.getElementById("btn-add-example").disabled = !hasText || !hasHand;
  });

  document.getElementById("skip-frames").addEventListener("change", (e) => {
    skipFrames = e.target.checked;
  });

  document.getElementById("btn-export").addEventListener("click", exportJSON);
  document.getElementById("btn-import").addEventListener("click", () => {
    document.getElementById("import-file").click();
  });
  document.getElementById("import-file").addEventListener("change", importJSON);
  document.getElementById("btn-clear-all").addEventListener("click", clearAllSigns);
}

function setMode(predict) {
  isPredictMode = predict;
  document.getElementById("btn-predict-mode").classList.toggle("active", predict);
  document.getElementById("btn-teach-mode").classList.toggle("active", !predict);
  document.getElementById("predict-panel").classList.toggle("hidden", !predict);
  document.getElementById("teach-panel").classList.toggle("hidden", predict);
  document.getElementById("mode-display").textContent = predict ? "ทำนาย" : "สอนท่า";

  if (!predict) {
    document.getElementById("prediction-display").textContent = "–";
    document.getElementById("confidence-display").textContent = "–";
  }
}

function updateHandStatus(found) {
  const el = document.getElementById("hand-status");
  el.textContent = found ? "พบมือ" : "ยังไม่พบ";
  el.style.color = found ? "var(--success)" : "var(--text-muted)";

  const label = document.getElementById("sign-label").value.trim();
  document.getElementById("btn-add-example").disabled = !found || !label;
}

function addExample() {
  const labelInput = document.getElementById("sign-label");
  const label = labelInput.value.trim();

  if (!label) {
    alert("กรุณาใส่ชื่อท่ามือก่อน");
    return;
  }
  if (!currentHandFeatures) {
    alert("ยังจับมือไม่เจอ – ลองปรับมุมกล้องหรือแสง");
    return;
  }

  knnClassifier.addExample(currentHandFeatures, label);

  if (!signsData[label]) signsData[label] = [];
  signsData[label].push([...currentHandFeatures]);

  exampleCounts[label] = signsData[label].length;
  updateExampleCountDisplay(label);
  updateSignListUI();
  saveSignsToDB();
}

function updateExampleCountDisplay(currentLabel) {
  const count = exampleCounts[currentLabel] || 0;
  const el = document.getElementById("example-count");
  el.textContent = `ตัวอย่างของ "${currentLabel}": ${count} ครั้ง`;
  el.style.color = count < 10 ? "var(--warning)" : "var(--success)";
}

function updateSignListUI() {
  const list = document.getElementById("sign-list");
  const labels = Object.keys(exampleCounts);

  if (labels.length === 0) {
    list.innerHTML = '<li class="empty">ยังไม่มีท่าที่สอน</li>';
    return;
  }

  list.innerHTML = "";
  labels.sort().forEach((label) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span>${label}</span>
      <span>
        <span class="count">${exampleCounts[label]} ตัวอย่าง</span>
        <button class="delete-btn" data-label="${label}" title="ลบท่านี้">ลบ</button>
      </span>
    `;
    list.appendChild(li);
  });

  list.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const label = btn.dataset.label;
      if (confirm(`ลบท่า "${label}" ทั้งหมด?`)) {
        deleteSign(label);
      }
    });
  });
}

function deleteSign(labelToDelete) {
  delete signsData[labelToDelete];
  delete exampleCounts[labelToDelete];

  knnClassifier = ml5.KNNClassifier();
  for (const label of Object.keys(signsData)) {
    for (const feat of signsData[label]) {
      knnClassifier.addExample(feat, label);
    }
  }

  saveSignsToDB();
  updateSignListUI();
  document.getElementById("example-count").textContent = "ยังไม่มีตัวอย่าง";
}

function clearAllSigns() {
  if (!confirm("ลบท่าที่สอนไว้ทั้งหมดจริงหรือไม่? การกระทำนี้ย้อนกลับไม่ได้")) return;

  knnClassifier = ml5.KNNClassifier();
  signsData = {};
  exampleCounts = {};
  clearDB();
  updateSignListUI();
  document.getElementById("example-count").textContent = "ยังไม่มีตัวอย่าง";
  document.getElementById("prediction-display").textContent = "–";
  document.getElementById("confidence-display").textContent = "–";
}

// ==================== IndexedDB ====================
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };
    request.onupgradeneeded = (event) => {
      const database = event.target.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "label" });
      }
    };
  });
}

function saveSignsToDB() {
  if (!db) return;
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);
  store.clear();
  for (const label of Object.keys(signsData)) {
    store.put({ label, features: signsData[label] });
  }
}

function loadSignsFromDB() {
  return new Promise((resolve) => {
    if (!db) { resolve(); return; }
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();
    request.onsuccess = () => {
      const rows = request.result || [];
      signsData = {};
      exampleCounts = {};
      knnClassifier = ml5.KNNClassifier();
      for (const row of rows) {
        signsData[row.label] = row.features;
        exampleCounts[row.label] = row.features.length;
        for (const feat of row.features) {
          knnClassifier.addExample(feat, row.label);
        }
      }
      updateSignListUI();
      resolve();
    };
    request.onerror = () => resolve();
  });
}

function clearDB() {
  if (!db) return;
  const tx = db.transaction(STORE_NAME, "readwrite");
  tx.objectStore(STORE_NAME).clear();
}

// ==================== Export / Import ====================
function exportJSON() {
  const payload = {
    signs: Object.keys(signsData).map((label) => ({
      label,
      features: signsData[label]
    }))
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sign-pose-data-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function importJSON(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      if (!data.signs || !Array.isArray(data.signs)) {
        alert("ไฟล์ JSON ไม่ถูกต้อง (ต้องมี key \"signs\" เป็น array)");
        return;
      }
      knnClassifier = ml5.KNNClassifier();
      signsData = {};
      exampleCounts = {};
      for (const sign of data.signs) {
        if (!sign.label || !Array.isArray(sign.features)) continue;
        signsData[sign.label] = sign.features;
        exampleCounts[sign.label] = sign.features.length;
        for (const feat of sign.features) {
          knnClassifier.addExample(feat, sign.label);
        }
      }
      saveSignsToDB();
      updateSignListUI();
      alert(`นำเข้าสำเร็จ ${Object.keys(signsData).length} ท่า`);
    } catch (err) {
      console.error(err);
      alert("อ่านไฟล์ JSON ไม่สำเร็จ");
    }
  };
  reader.readAsText(file);
  event.target.value = "";
}

// ==================== เริ่มทำงาน ====================
window.addEventListener("DOMContentLoaded", init);
