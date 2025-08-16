// server.js - Improved OCR endpoint with better error handling
import dotenv from "dotenv";
dotenv.config(); // Load environment variables from .env file

import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import Tesseract from "tesseract.js";
import { createClerkClient } from "@clerk/backend";
import { GoogleGenerativeAI } from "@google/generative-ai";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const currentDirname = path.dirname(__filename);

// --- Multer Configuration ---
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(currentDirname, "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(
      null,
      `${file.fieldname}-${Date.now()}${path.extname(file.originalname)}`
    );
  },
});

// Add file filter to only accept images
const fileFilter = (req, file, cb) => {
  const allowedMimeTypes = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
  ];
  if (allowedMimeTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error("Invalid file type. Only image files are allowed."), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
});

// Check for required environment variables
const requiredEnvVars = ["GEMINI_API_KEY"];
const missingEnvVars = requiredEnvVars.filter((envVar) => !process.env[envVar]);

if (missingEnvVars.length > 0) {
  console.error("❌ Missing required environment variables:", missingEnvVars);
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

const app = express();
const PORT = process.env.PORT || 3000;

const clerkClient = createClerkClient({
  secretKey: process.env.CLERK_SECRET_KEY,
});

app.use(cors());
app.use(bodyParser.json());

// Enhanced OCR endpoint with better error handling
app.post("/ocr", (req, res) => {
  upload.single("image")(req, res, async (uploadErr) => {
    // ... (Your existing Multer error handling) ...
    if (uploadErr) {
      console.error("❌ Multer error:", uploadErr.message);
      return res.status(400).json({
        error: "File upload failed",
        details: uploadErr.message,
      });
    }

    let worker = null;
    let filePath = null;

    try {
      if (!req.file) {
        return res.status(400).json({
          error:
            "No image file uploaded. Make sure to use 'image' as the form field name.",
        });
      }

      filePath = path.resolve(req.file.path);
      console.log("🖼️ Processing file:", filePath);

      if (!fs.existsSync(filePath)) {
        throw new Error("Uploaded file not found on server");
      }

      // --- Streamlined Tesseract Initialization ---
      console.log("🔧 Initializing Tesseract worker for Turkish OCR...");

      // Tesseract.js v5+ requires a specific language code.
      // We will only try to initialize with 'tur' as that is the user's intent.
      // The library will automatically download the language data if needed.
      worker = await Tesseract.createWorker("tur");
      console.log(
        "✅ Tesseract worker initialized with Turkish language model"
      );

      // --- Tesseract Recognition ---
      console.log("🔍 Performing OCR on the image...");
      const result = await worker.recognize(filePath);
      const rawText = result.data.text.trim();

      if (!rawText) {
        throw new Error("No text detected in the image");
      }

      console.log("📄 OCR completed, text length:", rawText.length);

      // --- Cleanup ---
      await worker.terminate();
      worker = null;
      fs.unlinkSync(filePath);
      filePath = null;

      // ... (Rest of your Gemini processing code) ...
      if (rawText.length < 10) {
        return res.json({
          ocr_result: {
            corrected_text: rawText,
            yks_topics: [],
            note: "Text too short to categorize",
          },
          raw_text: rawText,
        });
      }

      console.log("🤖 Processing with Gemini...");
      const prompt = `OCR ile bir fotoğraftan aşağıdaki yks sorusunu çıkardım, ocr sisteminden kaynaklı Yazım ve mantık hatalarını gider, metni düzeltilmiş haliyle JSON çıktısının "corrected_text" alanına ekleyin. Ayrıca, sorunun ait olduğu YKS (Yükseköğretim Kurumları Sınavı) konularını "yks_topics" alanına listeyin (örneğin: "TYT-Biyologi-Bitkiler", "AYT-Kimya-Asitler-Bazlar"). Soru çözümünü kesinlikse vermeyin.

Metin:
${rawText}`;

      const payload = {
        contents: [
          {
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: {
            type: "OBJECT",
            properties: {
              corrected_text: { type: "STRING" },
              yks_topics: {
                type: "ARRAY",
                items: { type: "STRING" },
              },
              note: { type: "STRING" },
            },
            required: ["corrected_text", "yks_topics"],
          },
        },
      };

      // ... (Rest of your Gemini API call and response handling) ...
      console.log("📤 Sending request to Gemini API...");
      let geminiResponse;
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);
        geminiResponse = await Promise.race([
          model.generateContent(payload),
          new Promise((_, reject) =>
            setTimeout(
              () => reject(new Error("Gemini API timeout after 30 seconds")),
              30000
            )
          ),
        ]);
        clearTimeout(timeoutId);
        console.log("📥 Received response from Gemini API");
      } catch (geminiError) {
        // ... (Your existing Gemini error handling) ...
        console.error("❌ Gemini API Error:", geminiError);
        console.log("⚠️ Returning OCR-only result due to Gemini failure");
        return res.json({
          ocr_result: {
            corrected_text: rawText,
            yks_topics: [],
            note: "AI processing unavailable - raw OCR result returned",
          },
          raw_text: rawText,
          success: true,
          warning: "AI processing failed, OCR completed successfully",
        });
      }

      // ... (Rest of your response parsing and final JSON response) ...
      let structuredData;
      try {
        const responseText =
          geminiResponse.response.candidates[0].content.parts[0].text;
        structuredData = JSON.parse(responseText);
      } catch (parseErr) {
        console.error(
          "❌ Failed to parse Gemini response as JSON:",
          parseErr.message
        );
        structuredData = {
          corrected_text: rawText,
          yks_topics: [],
          note: "Failed to process with AI, returning raw OCR result",
        };
      }

      console.log("✅ Processing completed successfully");
      res.json({
        ocr_result: structuredData,
        raw_text: rawText,
        success: true,
      });
    } catch (err) {
      // ... (Your existing error cleanup and response) ...
      console.error("❌ OCR endpoint error:", err);
      if (worker) {
        try {
          await worker.terminate();
        } catch (workerErr) {
          console.error("❌ Error terminating worker:", workerErr);
        }
      }
      if (filePath && fs.existsSync(filePath)) {
        try {
          fs.unlinkSync(filePath);
        } catch (unlinkErr) {
          console.error("❌ Error deleting file:", unlinkErr);
        }
      }
      const errorResponse = {
        error: "OCR processing failed",
        details: err.message,
        success: false,
      };
      const statusCode = err.message.includes("No text detected") ? 422 : 500;
      res.status(statusCode).json(errorResponse);
    }
  });
});
// Debug endpoint to check environment variables
app.get("/debug-env", (req, res) => {
  res.json({
    hasGeminiKey: !!process.env.GEMINI_API_KEY,
    geminiKeyLength: process.env.GEMINI_API_KEY
      ? process.env.GEMINI_API_KEY.length
      : 0,
    geminiKeyPrefix: process.env.GEMINI_API_KEY
      ? process.env.GEMINI_API_KEY.substring(0, 10) + "..."
      : "not found",
    nodeEnv: process.env.NODE_ENV,
    port: process.env.PORT,
  });
});

// Test endpoint to verify Gemini API is working
app.post("/test-gemini", async (req, res) => {
  try {
    console.log("🧪 Testing Gemini API connection...");

    if (!process.env.GEMINI_API_KEY) {
      return res.status(400).json({
        error: "GEMINI_API_KEY not found in environment variables",
        success: false,
      });
    }

    const testPrompt =
      "Say 'Hello, Gemini API is working!' in JSON format with a 'message' field.";

    const payload = {
      contents: [
        {
          parts: [{ text: testPrompt }],
        },
      ],
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema: {
          type: "OBJECT",
          properties: {
            message: { type: "STRING" },
          },
          required: ["message"],
        },
      },
    };

    console.log("📤 Sending test request to Gemini...");
    const response = await model.generateContent(payload);

    console.log("📥 Received response structure:", Object.keys(response));

    // Handle the response structure - it might have a 'response' wrapper
    let candidates;
    if (response.response && response.response.candidates) {
      candidates = response.response.candidates;
      console.log("✅ Found candidates in response.response");
    } else if (response.candidates) {
      candidates = response.candidates;
      console.log("✅ Found candidates directly");
    } else {
      console.log("❌ No candidates found in response:", response);
      return res.status(500).json({
        error: "No candidates found in response",
        response: response,
        success: false,
      });
    }

    if (
      candidates &&
      candidates[0] &&
      candidates[0].content &&
      candidates[0].content.parts &&
      candidates[0].content.parts[0]
    ) {
      const textResponse = candidates[0].content.parts[0].text;
      const result = JSON.parse(textResponse);
      res.json({
        success: true,
        gemini_response: result,
        message: "Gemini API is working correctly!",
        raw_response: textResponse,
      });
    } else {
      res.status(500).json({
        error: "Invalid response structure from Gemini",
        response: response,
        success: false,
      });
    }
  } catch (err) {
    console.error("❌ Gemini test error:", err);
    res.status(500).json({
      error: "Gemini API test failed",
      details: err.message,
      success: false,
    });
  }
});

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({ status: "OK", timestamp: new Date().toISOString() });
});

// Test endpoint to verify server is running
app.get("/", (req, res) => {
  res.json({
    message: "OCR Server is running",
    endpoints: ["/ocr", "/health", "/signup", "/signin"],
  });
});
//signup
app.post("/signup", async (req, res) => {
  const { email, password, firstName, lastName } = req.body;

  try {
    const user = await clerk.users.createUser({
      emailAddress: [email],
      password,
      firstName,
      lastName,
    });

    res.json({ success: true, user });
  } catch (err) {
    res.status(400).json({ success: false, error: err.errors || err.message });
  }
});

app.post("/signin", async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password)
    return res.status(400).json({ error: "Missing email or password" });
  try {
    const signInToken = await clerkClient.signInTokens.create({
      identity: { emailAddress: [email] },
    });
    res.json(signInToken);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.errors || err.message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📋 Available endpoints:`);
  console.log(`   GET  /          - Server info`);
  console.log(`   GET  /health    - Health check`);
  console.log(`   POST /ocr       - OCR processing`);
  console.log(`   POST /signup    - User signup`);
  console.log(`   POST /signin    - User signin`);
});
