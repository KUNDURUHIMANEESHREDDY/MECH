/**
 * Research Package Manager (.interp) Service.
 */

export class ResearchPackageManager {
  constructor() {
    this.packages = [
      { id: 'pkg_ioi_suite', name: 'ioi-circuit-toolkit', version: '2.1.0', author: 'AI Safety Lab' },
      { id: 'pkg_sae_lens', name: 'sae-polysemanticity-lens', version: '1.4.0', author: 'InterpLab' },
    ];
  }

  createPackage(packageId, packageName) {
    const newPkg = { id: packageId, name: packageName, version: '1.0.0', status: 'Published' };
    this.packages.push(newPkg);
    return newPkg;
  }

  installPackage(packageName) {
    return {
      packageName,
      status: 'Installed',
      installedAt: new Date().toISOString(),
    };
  }

  listPackages() {
    return [...this.packages];
  }
}

export const researchPackageManager = new ResearchPackageManager();
