plugins {
    id("com.android.application")
    kotlin("android")
}
android {
    namespace = "vn.childsafe.app"
    compileSdk = 35
    defaultConfig {
        applicationId = "vn.childsafe.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "0.1.0-test02"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("onnx", "task") }
}
dependencies {
    implementation(project(":core"))
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.lifecycle:lifecycle-service:2.9.0")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.google.zxing:core:3.5.3")
    implementation("com.google.mediapipe:tasks-vision:0.10.21")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.21.0")
    testImplementation("junit:junit:4.13.2")
}
