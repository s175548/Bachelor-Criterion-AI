
    self.binary_class_dictionary = self.generate_binary_class_dictionary()


def generate_binary_class_dictionary(self):
        """     All categories found in metadata_csv are turned into dictionary, such that that can get a binary output (0: good, 1: defect) by parsing the category to the dict
                self.binary_class_dictionary[ self.metadata_csv[0,0] ] will return the binary value of the first datapoint.
        """
        binary_dict = {}
        for ele in np.unique(self.metadata_csv[:, 0]):
            if "good" in ele.lower():
                binary_dict[ele] = 0
            else:
                binary_dict[ele] = 1
        return binary_dict



    def read_segmentation_file(self,filename):
        """     Helper function, that simply opens segmentation file, draws a contour from this.
                Output: Segmentation retrieved from filename
        """
        fh = open(filename, "r")
        try:
            file_content = fh.read()
            seg = json.loads(file_content)
            segmentation = draw_contours2(seg, label_space={kk["label"]: [1.0] for kk in seg["annotations"]})
            return segmentation
        finally:
            fh.close()

            np.where(np.array([numb in dataloader.valid_annotations for numb in list(range(691))]) == False)[0] # Get index of invalid masks
            np.where(np.array(dataloader.visibility_score) == 3)[0] #
            np.sort(np.array(list(np.where(np.array(dataloader.visibility_score) == 3)[0]) + list(
                np.where(np.array(dataloader.visibility_score) == 2)[0])))