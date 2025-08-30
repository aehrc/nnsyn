import json
import unittest


class Test_TestPreprocessing(unittest.TestCase):
    def setUp(self):
        file1 = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset260_synthrad2025_task1_MR_AB_pre_v2r_stitched_masked/gt_nnUNetResEncUNetLPlans.json"
        file2 = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset960_synthrad2025_task1_mri2ct_AB/gt_plan/nnUNetResEncUNetLPlans.json"
        with open(file1, 'r') as f:
            self.data1 = json.load(f)
        with open(file2, 'r') as f:
            self.data2 = json.load(f)

    def test_gt_intensity_mean(self):
        self.assertAlmostEqual(self.data1['foreground_intensity_properties_per_channel']['0']['mean'],
                               self.data2['foreground_intensity_properties_per_channel']['0']['mean'], delta=0.1)
    def test_gt_intensity_std(self):
        self.assertAlmostEqual(self.data1['foreground_intensity_properties_per_channel']['0']['std'],
                               self.data2['foreground_intensity_properties_per_channel']['0']['std'], delta=0.1)
    def test_plans_name(self):
        self.assertEqual(self.data1['plans_name'], self.data2['plans_name'])
    
    def test_normalization_schemes(self):
        self.assertEqual(self.data1['configurations']['3d_fullres']['normalization_schemes'], self.data2['configurations']['3d_fullres']['normalization_schemes'])


if __name__ == "__main__":
    unittest.main()