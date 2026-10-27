pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "ChildSafeAndroid"
include(":core")
// Allows protocol/alarm tests without an Android SDK.
if (providers.gradleProperty("coreOnly").orNull != "true") include(":app")
