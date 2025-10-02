@echo off
echo Preparing FastAPI app for Render deployment...

echo.
echo Step 1: Initialize Git repository
git init

echo.
echo Step 2: Add all files
git add .

echo.
echo Step 3: Commit changes
git commit -m "Deploy FastAPI placement prediction app to Render"

echo.
echo Step 4: Set main branch
git branch -M main

echo.
echo ========================================
echo Git repository prepared successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Create a GitHub repository
echo 2. Add remote: git remote add origin YOUR_GITHUB_URL
echo 3. Push: git push -u origin main
echo 4. Deploy on Render.com using your GitHub repo
echo.
echo OR upload the project folder directly to Render.com
echo ========================================

pause